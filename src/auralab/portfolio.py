from dataclasses import dataclass
from functools import cached_property
from datetime import timedelta
import torch
from torch import Tensor


@dataclass(frozen=True)
class PortfolioStats:
    """
    Calculates portfolio statistics and returns for optimization.

    Args:
        log_price: Tensor of shape (T, Assets) containing log prices.
        weights: Tensor of shape (T, Assets) containing portfolio weights.
                 Values should be in [-1, 1] for long/short.
        cost_bps: Transaction cost in basis points (e.g. 0.0010 for 10bps).
    """

    log_price: Tensor
    weights: Tensor
    cost_bps: float
    t_dim: int = 0
    a_dim: int = -1
    sample_rate: timedelta = timedelta(days=1)

    @property
    def steps_per_year(self) -> float:
        """Number of steps in a year based on sampling period."""
        return timedelta(days=365.25) / self.sample_rate

    @cached_property
    def forward_returns(self) -> Tensor:
        """
        Forward returns: r_t = p_{t+1} - p_t
        The last element is 0 because there is no p_{t+1}.
        """
        # Calculate diff along time dimension
        # append=last_val ensures shape matches log_price and last return is 0

        # Get the last value along t_dim and unsqueeze to keep dimensions
        last_val = self.log_price.select(self.t_dim, -1).unsqueeze(self.t_dim)

        return torch.diff(self.log_price, dim=self.t_dim, append=last_val)

    @cached_property
    def iso_gross_perf(self) -> Tensor:
        """
        Portfolio gross performance at time t: R_t = w_t * r_t

        Since r_t is the return from t to t+1, we use weights w_t
        (decided at t) to capture this return.
        """
        # Element-wise multiplication
        # shape: (T, Assets)
        return self.weights * self.forward_returns

    @cached_property
    def iso_log_cost(self) -> Tensor:
        """
        Transaction costs incurred at time t, in log-space.
        Cost fraction = bps * |w_t - w_{t-1}|
        Returns log(1 - cost_fraction) = log1p(-cost_fraction)
        """
        # Change in weights from previous step
        # We assume initial weights (before t=0) are 0.

        zeros_shape = list(self.weights.shape)
        zeros_shape[self.t_dim] = 1
        zeros = torch.zeros(
            zeros_shape, device=self.weights.device, dtype=self.weights.dtype
        )

        # diff(prepend=zeros) gives w_t - w_{t-1}
        # shape: (T, Assets)
        delta_w = torch.diff(self.weights, dim=self.t_dim, prepend=zeros).abs()
        linear_cost = delta_w * self.cost_bps
        return torch.log1p(-linear_cost)

    @cached_property
    def iso_net_perf(self) -> Tensor:
        """
        Net performance after transaction costs.
        Since we are in log-space:
        Net = Gross + Log(1 - Cost)
        """
        return self.iso_gross_perf + self.iso_log_cost

    @cached_property
    def iso_total_perf(self) -> Tensor:
        """Total performance over the period (sum of net performance)."""
        # Exclude the last step (which is 0 padding)
        net_ret = self.iso_net_perf.narrow(
            self.t_dim, 0, self.iso_net_perf.size(self.t_dim) - 1
        )
        return net_ret.sum(dim=self.t_dim)

    @cached_property
    def iso_annualized_perf(self) -> Tensor:
        """Annualized performance based on mean return per step."""
        # Exclude the last step (which is 0 padding)
        net_ret = self.iso_net_perf.narrow(
            self.t_dim, 0, self.iso_net_perf.size(self.t_dim) - 1
        )
        # Mean return per step * steps_per_year
        # shape: (Assets,)
        return net_ret.mean(dim=self.t_dim) * self.steps_per_year

    @cached_property
    def iso_sharpe(self) -> Tensor:
        """Annualized Sharpe ratio per asset"""
        # Exclude the last step (which is 0 padding)
        net_ret = self.iso_net_perf.narrow(
            self.t_dim, 0, self.iso_net_perf.size(self.t_dim) - 1
        )

        # Avoid division by zero
        # shape: (Assets,)
        std = net_ret.std(dim=self.t_dim)
        mean = net_ret.mean(dim=self.t_dim)

        # Handle zero std
        sharpe = (mean / std) * (self.steps_per_year**0.5)
        return torch.nan_to_num(sharpe, nan=0.0)

    def _adjust_dim(self, dim: int) -> int:
        """Adjust dimension index after t_dim reduction."""
        ndim = self.weights.ndim
        if dim < 0:
            dim += ndim
        t_dim = self.t_dim
        if t_dim < 0:
            t_dim += ndim

        if dim < t_dim:
            return dim
        elif dim > t_dim:
            return dim - 1
        else:
            raise ValueError("Cannot sum over t_dim in reduced stats")

    @cached_property
    def max_perf_loss(self) -> Tensor:
        """
        Loss function for minimization.
        We want to maximize return, so minimize negative annualized performance.
        Aggregates over assets.
        """
        return -self.iso_annualized_perf.sum(dim=self._adjust_dim(self.a_dim))

    @cached_property
    def cross_net_perf(self) -> Tensor:
        """
        Net performance of the entire portfolio at each time step.
        Sum of net performance across assets.
        Keeps dimensions to preserve t_dim index for subsequent calculations.
        """
        # Sum over asset dimension, keepdim=True to preserve t_dim
        return self.iso_net_perf.sum(dim=self.a_dim, keepdim=True)

    @cached_property
    def cross_sharpe(self) -> Tensor:
        """
        Annualized Sharpe ratio of the portfolio.
        Calculated on the aggregated portfolio performance.
        """
        # Calculate stats on the portfolio net return series
        # Use t_dim from the original stats (preserved in cross_net_perf)
        # Exclude the last step (which is 0 padding)
        r = self.cross_net_perf.narrow(
            self.t_dim, 0, self.cross_net_perf.size(self.t_dim) - 1
        )

        std = r.std(dim=self.t_dim)
        mean = r.mean(dim=self.t_dim)

        # Handle zero std
        sharpe = (mean / std) * (self.steps_per_year**0.5)
        return torch.nan_to_num(sharpe, nan=0.0)

    @cached_property
    def cross_loss_sharpe(self) -> Tensor:
        """
        Loss function for maximizing Portfolio Sharpe ratio.
        Minimizes negative Portfolio Sharpe ratio.
        """
        return -self.cross_sharpe.sum()

    def hybrid_loss(
        self,
        return_target: float | None = None,
        sharpe_target: float | None = None,
        penalty_weight: float = 100.0,
    ) -> Tensor:
        """
        Dynamic Hybrid loss function.

        Objective: Maximize Return + Dynamic Sharpe Bonus.
        Constraints: Penalize if Sharpe < Target.

        Logic:
            1. Excess Return = ReLU(Return - Return_Target) (if return_target is not None)
            2. Sharpe Deficit = ReLU(Sharpe_Target - Sharpe) (if sharpe_target is not None)
            3. Objective = Return + Excess_Return * Sharpe
            4. Penalty = Penalty_Weight * Sharpe_Deficit^2

        Loss = -Objective + Penalty

        Args:
            return_target: Threshold for return to activate Sharpe bonus. If None, bonus is disabled.
            sharpe_target: Minimum target for Sharpe ratio. If None, penalty is disabled.
            penalty_weight: Weight for Sharpe penalty.
        """
        # Calculate Portfolio Annualized Return
        # cross_net_perf is (T, 1) or (Batch, T, 1)
        r = self.cross_net_perf.narrow(
            self.t_dim, 0, self.cross_net_perf.size(self.t_dim) - 1
        )
        port_annualized_ret = r.mean(dim=self.t_dim) * self.steps_per_year
        # shape: (Batch, 1) or (1,) if squeezed. cross_sharpe is (Batch,) or scalar.

        # Ensure shapes match for addition/comparison
        if port_annualized_ret.ndim > self.cross_sharpe.ndim:
            port_annualized_ret = port_annualized_ret.squeeze(-1)

        # 1. Dynamic Objective
        objective = port_annualized_ret

        if return_target is not None:
            # Excess Return: How much are we above the return target?
            excess_return = torch.relu(port_annualized_ret - return_target)
            # Add Sharpe bonus scaled by excess return.
            objective = objective + excess_return * self.cross_sharpe

        # 2. Penalties
        penalty = torch.tensor(0.0, device=self.log_price.device)

        if sharpe_target is not None:
            # Sharpe Deficit: How much are we below the Sharpe target?
            sharpe_deficit = torch.relu(sharpe_target - self.cross_sharpe)
            # Penalty grows quadratically with deficit
            penalty = penalty_weight * sharpe_deficit.pow(2)

        # Total Loss (Minimize negative objective + penalty)
        return (-objective + penalty).sum()


def portfolio(
    log_price: Tensor,
    weights: Tensor,
    cost_bps: float = 0.0010,
    t_dim: int = 0,
    a_dim: int = -1,
    sample_rate: timedelta = timedelta(days=1),
) -> PortfolioStats:
    return PortfolioStats(log_price, weights, cost_bps, t_dim, a_dim, sample_rate)


def signed_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Signed Softmax activation.

    Applies softmax to the absolute values of the input, then restores the original signs.
    The result satisfies sum(abs(output)) == 1 along the specified dimension.

    Args:
        x: Input tensor.
        dim: Dimension along which to apply softmax.

    Returns:
        Tensor with same shape as x.
    """
    # Softmax on absolute values
    s = torch.softmax(x.abs(), dim=dim)
    # Restore signs
    return x.sign() * s
