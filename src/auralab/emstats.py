from typing import Optional
from torch import Tensor
import torch
from dataclasses import dataclass
from functools import cached_property, cache
from auralab.ema import ema


@dataclass(frozen=True)
class EMStats:
    val: Tensor
    alpha: Tensor
    dim: int = 0
    lookahead: int = 2

    @dataclass(frozen=True)
    class State:
        avg: Tensor
        var: Tensor
        cov_triu: tuple[tuple[int, Tensor], ...]  # by dim_c

    initial_state: Optional[State] = None

    @cached_property
    def _cov_triu_states(self) -> dict[int, Tensor]:
        """Mutable container for exx_triu states."""
        return {}

    @property
    def final_state(self) -> State:
        # Collect last values
        # avg
        avg_last = self.avg.select(self.dim, -1)
        # var
        var_last = self.var.select(self.dim, -1)
        # exx_triu
        cov_triu_last = tuple(
            sorted(
                ((k, v.select(self.dim, -1)) for k, v in self._cov_triu_states.items()),
                key=lambda x: x[0],
            )
        )
        return self.State(avg=avg_last, var=var_last, cov_triu=cov_triu_last)

    @cached_property
    def avg(self) -> Tensor:
        """Moving average (EMA)."""
        state = self.initial_state.avg if self.initial_state else None
        return ema(
            self.val, self.alpha, state=state, dim=self.dim, lookahead=self.lookahead
        )

    @property
    def _val_expanded(self) -> Tensor:
        # Expand val to match avg shape: (*val.shape, *alpha.shape)
        # We need to add alpha.ndim singleton dimensions at the end of val
        return self.val.view(*self.val.shape, *([1] * self.alpha.ndim))

    @cached_property
    def var(self) -> Tensor:
        """Moving variance.
        Var(X) = E[X^2] - (E[X])^2
        """
        # Var(X) = E[(X - E[X])^2]
        # This is more numerically stable than E[X^2] - (E[X])^2
        state = None
        if self.initial_state:
            # State for centered variance is just the variance itself
            state = self.initial_state.var

        # Use the already computed avg
        # We need to expand avg to match val shape if needed, but _val_expanded handles val.
        # avg has shape (*val.shape, *alpha.shape)
        # val has shape (*val.shape)
        # _val_expanded has shape (*val.shape, *alpha.shape)

        diff_sq = (self._val_expanded - self.avg) ** 2

        # Use distributed=True if alpha is not scalar
        distributed = self.alpha.ndim > 0

        x2_avg = ema(
            diff_sq,
            self.alpha,
            state=state,
            dim=self.dim,
            lookahead=self.lookahead,
            distributed=distributed,
        )
        return x2_avg

    @cached_property
    def rstd(self) -> Tensor:
        """Reciprocal of moving standard deviation (1/std).
        Computed using rsqrt for numerical stability and performance.
        Variance is clamped to 1e-16 (std ~ 1e-8) to avoid division by zero.
        """
        return torch.rsqrt(self.var.clamp(min=1e-16))

    @cached_property
    def std(self) -> Tensor:
        return 1 / self.rstd

    @cached_property
    def zscore(self) -> Tensor:
        """Standardized score (Z-Score)."""
        return (self._val_expanded - self.avg) * self.rstd

    @cache
    def cov(self, dim_c: int) -> Tensor:
        """
        Moving covariance matrix between components along dim_c.
        Returns the strictly upper triangular part flattened.

        Args:
            dim_c: The dimension representing the components (features).

        Returns:
            Tensor of shape (..., N_pairs, ..., A)
            where N_pairs = C*(C-1)/2
        """
        # We need to compute E[X_i * X_j] - E[X_i] * E[X_j] for i <= j (or i < j)

        x_in = self.val
        if dim_c < 0:
            dim_c += x_in.ndim

        C = x_in.shape[dim_c]

        # Get indices for upper triangle (strictly upper)
        offset = 1
        row_idx, col_idx = torch.triu_indices(C, C, offset=offset, device=x_in.device)

        # Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
        # This is more numerically stable.

        # Get means
        mu = self.avg
        mu_row = mu.index_select(dim_c, row_idx)
        mu_col = mu.index_select(dim_c, col_idx)

        # Get values (expanded)
        val_expanded = self._val_expanded
        x_row = val_expanded.index_select(dim_c, row_idx)
        x_col = val_expanded.index_select(dim_c, col_idx)

        # Centered product
        centered_prod = (x_row - mu_row) * (x_col - mu_col)

        target_dim = self.dim
        state = None
        state = None
        if self.initial_state:
            # Convert tuple to dict for lookup
            cov_triu_dict = dict(self.initial_state.cov_triu)
            if dim_c in cov_triu_dict:
                state = cov_triu_dict[dim_c]

        distributed = self.alpha.ndim > 0

        cov_triu = ema(
            centered_prod,
            self.alpha,
            state=state,
            dim=target_dim,
            lookahead=self.lookahead,
            distributed=distributed,
        )

        # Store for final_state
        # Store for final_state
        self._cov_triu_states[dim_c] = cov_triu

        return cov_triu

    @cache
    def corr(self, dim_c: int) -> Tensor:
        """
        Moving correlation matrix (normalized covariance).
        Returns the strictly upper triangular part flattened.
        R_ij = Cov_ij / (std_i * std_j)

        Args:
            dim_c: The dimension representing the components (features).

        Returns:
            Tensor of shape (..., N_pairs, ..., A)
            where N_pairs = C*(C-1)/2
        """
        # 1. Compute covariance matrix (flattened triu)
        cov_triu = self.cov(dim_c)

        # 2. Get reciprocal standard deviations
        rstd = self.rstd

        # 3. Construct rstd_i * rstd_j for the upper triangle
        x_in = self.val
        if dim_c < 0:
            dim_c += x_in.ndim

        C = x_in.shape[dim_c]
        offset = 1
        row_idx, col_idx = torch.triu_indices(C, C, offset=offset, device=x_in.device)

        # Note: self.var has same shape as x (plus alpha dim).
        # We need to select along dim_c.

        rstd_row = rstd.index_select(dim_c, row_idx)
        rstd_col = rstd.index_select(dim_c, col_idx)

        rstd_prod_triu = rstd_row * rstd_col

        return (cov_triu * rstd_prod_triu).clamp(-1.0, 1.0)


def emstats(
    val: Tensor,
    alpha: Tensor,
    dim: int = 0,
    lookahead: int = 2,
    initial_state: Optional[EMStats.State] = None,
) -> EMStats:
    return EMStats(val, alpha, dim, lookahead, initial_state)
