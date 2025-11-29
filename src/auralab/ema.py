import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def ema_scan_op(a_curr, b_curr, a_prev, b_prev):
    # Composition of linear functions:
    # f_curr(y) = a_curr * y + b_curr
    # f_prev(y) = a_prev * y + b_prev
    # (f_curr o f_prev)(y) = a_curr * (a_prev * y + b_prev) + b_curr
    #                      = (a_curr * a_prev) * y + (a_curr * b_prev + b_curr)
    return a_prev * a_curr, a_prev * b_curr + b_prev


@triton.jit
def ema_kernel(
    x_ptr,
    y_ptr,
    alpha_ptr,
    state_ptr,
    stride_x_batch,
    stride_x_time,
    stride_y_batch,
    stride_y_time,
    stride_y_alpha,
    stride_alpha,
    stride_state_batch,
    stride_state_alpha,
    n_elements,
    n_alphas,
    HAS_STATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DISTRIBUTED: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # pid corresponds to a unique (batch_idx, alpha_idx) pair
    # Grid is (B * A,)
    if DISTRIBUTED:
        # In distributed mode, we assume 1-to-1 mapping between sequence and alpha
        # pid identifies the unique sequence-alpha pair
        # alpha repeats every n_alphas
        alpha_idx = pid % n_alphas
        batch_idx = pid
    else:
        alpha_idx = pid % n_alphas
        batch_idx = pid // n_alphas

    # Pointers
    x_base = x_ptr + batch_idx * stride_x_batch
    y_base = y_ptr + batch_idx * stride_y_batch + alpha_idx * stride_y_alpha

    # Load alpha
    alpha_val = tl.load(alpha_ptr + alpha_idx * stride_alpha)

    # Load state if present
    prev_y = 0.0
    if HAS_STATE:
        prev_y = tl.load(
            state_ptr + batch_idx * stride_state_batch + alpha_idx * stride_state_alpha
        )

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x
    x = tl.load(x_base + offsets * stride_x_time, mask=mask, other=0.0)

    # Coefficients
    # y_t = (1 - alpha) * y_{t-1} + alpha * x_t
    a = tl.full([BLOCK_SIZE], 1.0 - alpha_val, dtype=tl.float32)
    b = alpha_val * x

    # If we have state, incorporate it into the first element
    if HAS_STATE:
        # b_0' = b_0 + a_0 * prev_y
        is_first = offsets == 0
        b = tl.where(is_first, b + a * prev_y, b)

    # Associative scan
    acc_a, acc_b = tl.associative_scan((a, b), 0, ema_scan_op)

    # Store result
    tl.store(y_base + offsets * stride_y_time, acc_b, mask=mask)


def ema_ref(
    x: torch.Tensor,
    alpha: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    dim: int = 0,
    distributed: bool = False,
) -> torch.Tensor:
    """Reference implementation using PyTorch loops."""
    # Move dim to 0 for easier iteration
    if dim != 0:
        x = x.transpose(dim, 0)

    # x is (T, ...)
    T = x.shape[0]

    # alpha is (A,)
    # We want output (T, ..., A) if we append A at the end?
    # Requirement: y.shape == [*x.shape, *alpha.shape]
    # If x is (T, B), y is (T, B, A).

    # Let's flatten x's other dims to B.
    # x -> (T, B)
    other_dims = x.shape[1:]
    x_flat = x.reshape(T, -1)
    B = x_flat.shape[1]

    # alpha -> (A,)
    alpha_flat = alpha.flatten()
    A = alpha_flat.numel()

    # Output (T, B, A)
    if distributed:
        # If distributed, we don't add A dimension.
        # x is (T, B) where B includes A.
        # We assume alpha broadcasts to B.
        # But wait, our B here is flattened other dims.
        # If distributed, alpha should match the last dims of x.
        # Here we flattened everything to B.
        # We need to ensure alpha broadcasts to B.
        # If x was (T, ..., A), then B = ... * A.
        # alpha is (A,).
        # We need to reshape alpha to (1, ..., A) and broadcast to (1, B).
        # Actually, let's just assume alpha is properly broadcastable to (B,)
        # if we handle it correctly.

        # But wait, in the loop:
        # curr = (1 - a) * curr + a * xt
        # xt is (B, 1).
        # a needs to be (B, 1).

        # If x is (T, N, A) and alpha is (A,).
        # x_flat is (T, N*A). B = N*A.
        # alpha needs to be repeated N times.
        # This is hard to do generically with flattened B.

        # Let's rely on the caller to ensure alpha matches if distributed?
        # Or we can just use the fact that alpha is (A,).
        # If distributed, we expect x.shape[-alpha.ndim:] == alpha.shape.

        # Let's not flatten alpha to (A,) if distributed.
        # Let's keep alpha as is, and reshape it to broadcast to B.
        pass

    y = torch.zeros(T, B, A if not distributed else 1, device=x.device, dtype=x.dtype)

    # State (B, A)
    curr = torch.zeros(B, A, device=x.device, dtype=x.dtype)
    if state is not None:
        curr = state.reshape(B, A)

    for t in range(T):
        # x_t is (B,)
        # we want to update curr (B, A)
        # x_t needs to be broadcast to (B, A)
        xt = x_flat[t].unsqueeze(-1)  # (B, 1)

        # curr = (1 - alpha) * curr + alpha * xt
        # alpha is (A,) -> (1, A)
        if distributed:
            # We need to construct 'a' of shape (B, 1)
            # alpha is (A,). B is multiple of A.
            # We can repeat alpha.
            # B = N * A.
            # alpha_flat is (A,).
            # We need (N*A,).
            n_repeats = B // A
            a = alpha_flat.repeat(n_repeats).unsqueeze(0).transpose(0, 1)  # (B, 1)

            # xt is (B, 1)
            # curr is (B, 1)
            curr = (1.0 - a) * curr + a * xt
            y[t] = curr
        else:
            a = alpha_flat.unsqueeze(0)
            curr = (1.0 - a) * curr + a * xt
            y[t] = curr

    # Reshape y
    # y is (T, B, A)
    # Restore B to other_dims
    y = y.reshape(T, *other_dims, A)

    # Restore dim
    if dim != 0:
        # We transposed dim to 0.
        # So T is at 0.
        # We want T at dim.
        # y currently: (T, d1, d2, ..., A)
        # We want: (d1, T, d2, ..., A) if dim=1
        # Just transpose(0, dim)
        y = y.transpose(0, dim)

    # Reshape A to alpha.shape
    if not distributed:
        y = y.reshape(*y.shape[:-1], *alpha.shape)
    else:
        # Remove the extra dimension we added (size 1)
        y = y.squeeze(-1)

    return y


def ema(
    x: torch.Tensor,
    alpha: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    dim: int = 0,
    lookahead: int = 1,
    optimized: bool = True,
    distributed: bool = False,
) -> torch.Tensor:
    """
    Computes EMA of x along dim using alphas.

    Args:
        x: Input tensor of shape (..., T, ...)
        alpha: Tensor of alphas (A,) or broadcastable to (A,)
        state: Optional initial state tensor.
               If provided, shape must match x's shape excluding dim, concatenated with alpha's shape.
               If None, the initial state is computed from x based on `lookahead`.
        dim: Dimension to scan along.
        lookahead: Number of samples to use for initialization when state is None.
                   If 1 (default), use the first sample.
                   If > 1, use the mean of the first `lookahead` samples.
        optimized: If True, use Triton kernel. If False, use PyTorch reference implementation.

        distributed: If True, alpha is applied element-wise to the last dimensions of x.
                     x.shape[-alpha.ndim:] must match alpha.shape.
                     Output shape will be x.shape.
                     If False (default), alpha is applied as a cross-product.
                     Output shape will be [*x.shape, *alpha.shape].

    Returns:
        y: Tensor of shape [*x.shape, *alpha.shape] (if not distributed)
           or x.shape (if distributed)
    """
    if distributed and alpha.ndim > 0:
        # Validate shapes
        if x.shape[-alpha.ndim :] != alpha.shape:
            raise ValueError(
                f"For distributed=True, x.shape[-alpha.ndim:] ({x.shape[-alpha.ndim:]}) "
                f"must match alpha.shape ({alpha.shape})"
            )
    if state is None:
        if lookahead == 1:
            # Initialize state with the first value of the sequence
            x0 = x.select(dim, 0)
        else:
            # Initialize state with the mean of the first `lookahead` values
            x0 = x.narrow(dim, 0, lookahead).mean(dim=dim)

        # Expand x0 to match the expected state shape: (*x_batch_dims, *alpha_dims)
        if alpha.ndim > 0:
            # We need to append alpha.ndim singleton dimensions to x0
            new_shape = x0.shape + (1,) * alpha.ndim
            x0 = x0.view(new_shape)
            # Expand to alpha shape
            target_shape = x0.shape[: -alpha.ndim] + alpha.shape
            state = x0.expand(target_shape)
        else:
            # alpha is scalar, state shape is just x0.shape
            state = x0

    if not optimized:
        return ema_ref(x, alpha, state, dim, distributed)

    assert x.is_cuda and alpha.is_cuda

    # Capture original shapes
    x_shape = x.shape
    alpha_shape = alpha.shape

    # Move dim to the end to simplify flattening
    if dim < 0:
        dim += x.ndim

    if dim != x.ndim - 1:
        x_in = x.transpose(dim, -1).contiguous()
    else:
        x_in = x.contiguous()

    B = x_in.numel() // x_in.shape[-1]
    T = x_in.shape[-1]

    x_flat = x_in.view(B, T)

    # Handle alpha
    alpha_flat = alpha.contiguous().flatten()
    A = alpha_flat.numel()

    # Output shape will be (B, T, A) temporarily
    y = torch.empty((B, T, A), device=x.device, dtype=x.dtype)

    # State
    if state is not None:
        # Expected state shape: (*x_batch_dims, *alpha_dims)
        # We flattened x_batch_dims to B, alpha_dims to A.
        state = state.contiguous().view(B, A if not distributed else 1)

    grid = (B * A,) if not distributed else (B,)

    MAX_BLOCK_SIZE = 65536

    if T <= MAX_BLOCK_SIZE:
        BLOCK_SIZE = triton.next_power_of_2(T)
        ema_kernel[grid](
            x_flat,
            y,
            alpha_flat,
            state,
            x_flat.stride(0),
            x_flat.stride(1),
            y.stride(0),
            y.stride(1),
            0 if distributed else y.stride(2),  # stride_y_alpha
            alpha_flat.stride(0),
            state.stride(0) if state is not None else 0,
            (
                0 if distributed else (state.stride(1) if state is not None else 0)
            ),  # stride_state_alpha
            T,
            A,
            HAS_STATE=state is not None,  # type: ignore
            BLOCK_SIZE=BLOCK_SIZE,
            DISTRIBUTED=distributed,  # type: ignore
        )
    else:
        # Chunking
        current_state = state
        for start in range(0, T, MAX_BLOCK_SIZE):
            end = min(start + MAX_BLOCK_SIZE, T)
            chunk_size = end - start
            BLOCK_SIZE = triton.next_power_of_2(chunk_size)

            x_chunk = x_flat[:, start:end]
            y_chunk = y[:, start:end, :]

            ema_kernel[grid](
                x_chunk,
                y_chunk,
                alpha_flat,
                current_state,
                x_flat.stride(0),
                x_flat.stride(1),
                y.stride(0),
                y.stride(1),
                0 if distributed else y.stride(2),
                alpha_flat.stride(0),
                current_state.stride(0) if current_state is not None else 0,
                (
                    0
                    if distributed
                    else (current_state.stride(1) if current_state is not None else 0)
                ),
                chunk_size,
                A,
                HAS_STATE=current_state is not None,  # type: ignore
                BLOCK_SIZE=BLOCK_SIZE,
                DISTRIBUTED=distributed,  # type: ignore
            )

            # Update state for next chunk
            # The state for the next chunk is the last element of the current chunk's output
            current_state = y_chunk[:, -1, :].contiguous()

    # Reshape y
    # y is (B, T, A).
    # We want to reshape B back to x_in's batch dims.
    # x_in was (*batch_dims, T).
    # So y becomes (*batch_dims, T, A).
    y = y.view(*x_in.shape[:-1], T, A)

    # Now we need to move T back to `dim`.
    # In x_in, T was at -1.
    # In y, T is at -2 (since A is at -1).
    # If we transposed x, we need to transpose y back.
    # We swapped `dim` and `-1` in x.
    # So we swap `dim` and `-2` in y.

    if dim != x.ndim - 1:
        y = y.transpose(dim, -2)

    # Finally reshape A to alpha_shape
    if not distributed:
        y = y.view(*y.shape[:-1], *alpha_shape)
    else:
        y = y.view(*y.shape[:-1])

    return y


def ema_decay(n: float | torch.Tensor) -> float | torch.Tensor:
    """
    Calculates the EMA alpha for a given span n.
    alpha = 2 / (n + 1)

    Args:
        n: The span (number of time steps).

    Returns:
        alpha: The decay factor.
    """
    return 2.0 / (n + 1.0)
