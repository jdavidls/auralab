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
        exx_triu: tuple[tuple[tuple[int, bool], Tensor], ...]  # by (dim_c, diagonal)

    initial_state: Optional[State] = None

    @cached_property
    def _exx_triu_states(self) -> dict[tuple[int, bool], Tensor]:
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
        exx_triu_last = tuple(
            sorted(
                ((k, v.select(self.dim, -1)) for k, v in self._exx_triu_states.items()),
                key=lambda x: x[0],
            )
        )
        return self.State(avg=avg_last, var=var_last, exx_triu=exx_triu_last)

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
        # E[X^2]
        state = None
        if self.initial_state:
            # Reconstruct E[X^2] state from Var and E[X]
            # E[X^2] = Var + (E[X])^2
            state = self.initial_state.var + self.initial_state.avg**2

        x2_avg = ema(
            self.val**2,
            self.alpha,
            state=state,
            dim=self.dim,
            lookahead=self.lookahead,
        )
        # (E[X])^2
        avg_x2 = self.avg**2
        return x2_avg - avg_x2

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
    def cov(self, dim_c: int, diagonal: bool = False) -> Tensor:
        """
        Moving covariance matrix between components along dim_c.
        Returns the upper triangular part flattened.

        Args:
            dim_c: The dimension representing the components (features).
            diagonal: If True, include diagonal elements (variances).
                      If False (default), exclude diagonal (strictly upper triangular).

        Returns:
            Tensor of shape (..., N_pairs, ..., A)
            where N_pairs = C*(C+1)/2 if diagonal=True
                          = C*(C-1)/2 if diagonal=False
        """
        # We need to compute E[X_i * X_j] - E[X_i] * E[X_j] for i <= j (or i < j)

        x_in = self.val
        if dim_c < 0:
            dim_c += x_in.ndim

        C = x_in.shape[dim_c]

        # Get indices for upper triangle
        offset = 0 if diagonal else 1
        row_idx, col_idx = torch.triu_indices(C, C, offset=offset, device=x_in.device)

        # Gather x values
        # x_in shape: (..., C, ...)
        # We need to select along dim_c

        # x_row: select row_idx along dim_c
        x_row = x_in.index_select(dim_c, row_idx)
        # x_col: select col_idx along dim_c
        x_col = x_in.index_select(dim_c, col_idx)

        # Compute product
        xx_prod = x_row * x_col

        # Compute EMA of the product
        # Note: dim_c is now the dimension of flattened pairs
        target_dim = self.dim
        # If we didn't change rank, target_dim stays same?
        # index_select preserves rank, just changes size of dim_c.
        # So target_dim is still valid unless it was dim_c.
        # But EMA scans along `dim` (time). `dim_c` is features.
        # So `dim` should be preserved.

        state = None
        if self.initial_state:
            # Convert tuple to dict for lookup
            exx_triu_dict = dict(self.initial_state.exx_triu)
            if (dim_c, diagonal) in exx_triu_dict:
                state = exx_triu_dict[(dim_c, diagonal)]

        exx_triu = ema(
            xx_prod,
            self.alpha,
            state=state,
            dim=target_dim,
            lookahead=self.lookahead,
        )

        # Store for final_state
        self._exx_triu_states[(dim_c, diagonal)] = exx_triu

        # Compute E[X_i] * E[X_j]
        mu = self.avg
        mu_row = mu.index_select(dim_c, row_idx)
        mu_col = mu.index_select(dim_c, col_idx)

        mu_prod_triu = mu_row * mu_col

        return exx_triu - mu_prod_triu

    @cache
    def corr(self, dim_c: int, diagonal: bool = False, eps: float = 1e-8) -> Tensor:
        """
        Moving correlation matrix (normalized covariance).
        Returns the upper triangular part flattened.
        R_ij = Cov_ij / (std_i * std_j)

        Args:
            dim_c: The dimension representing the components (features).
            diagonal: If True, include diagonal elements (always 1.0).
                      If False (default), exclude diagonal.
            eps: Small epsilon to avoid division by zero.

        Returns:
            Tensor of shape (..., N_pairs, ..., A)
            where N_pairs = C*(C+1)/2 if diagonal=True
                          = C*(C-1)/2 if diagonal=False
        """
        # 1. Compute covariance matrix (flattened triu)
        cov_triu = self.cov(dim_c, diagonal=diagonal)

        # 2. Get reciprocal standard deviations
        rstd = self.rstd

        # 3. Construct rstd_i * rstd_j for the upper triangle
        x_in = self.val
        if dim_c < 0:
            dim_c += x_in.ndim

        C = x_in.shape[dim_c]
        offset = 0 if diagonal else 1
        row_idx, col_idx = torch.triu_indices(C, C, offset=offset, device=x_in.device)

        # Note: self.var has same shape as x (plus alpha dim).
        # We need to select along dim_c.

        rstd_row = rstd.index_select(dim_c, row_idx)
        rstd_col = rstd.index_select(dim_c, col_idx)

        rstd_prod_triu = rstd_row * rstd_col

        return cov_triu * rstd_prod_triu


def emstats(
    val: Tensor,
    alpha: Tensor,
    dim: int = 0,
    lookahead: int = 2,
    initial_state: Optional[EMStats.State] = None,
) -> EMStats:
    return EMStats(val, alpha, dim, lookahead, initial_state)
