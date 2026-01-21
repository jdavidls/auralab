import torch
from neuralab.emstats import emstats


def inverse_zscore_step(z_t, alpha, mu_prev, var_prev):
    """
    Recover x_t from z_t and previous state.

    Formulas:
    sigma_t^2 = ((1 - alpha) * var_prev) / (1 - alpha * z_t^2)
    x_t = mu_prev + (z_t * sigma_t) / (1 - alpha)
    """
    denom = 1.0 - alpha * z_t**2
    # clamp denom to avoid division by zero or negative variance if z is too large
    # This implies alpha * z^2 < 1

    sigma2 = ((1.0 - alpha) * var_prev) / denom
    sigma = torch.sqrt(sigma2)

    x_t = mu_prev + (z_t * sigma) / (1.0 - alpha)

    # Update state for next step verification
    mu_curr = (1.0 - alpha) * mu_prev + alpha * x_t
    var_curr = sigma2  # Since var_curr IS sigma_t^2 in the EMStats formulation?
    # Wait, EMStats var is ema((x-mu)^2).
    # V_t = (1-a)V_{t-1} + a(x_t - mu_t)^2
    # We used this to derive sigma2. So yes, var_curr = sigma2.

    return x_t, mu_curr, var_curr


def verify():
    torch.manual_seed(42)
    T = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(T, device=device)
    alpha_val = 0.1
    alpha = torch.tensor([alpha_val], device=device)

    # Forward pass
    stats = emstats(x, alpha)
    z = stats.zscore

    # We need initial state (mu_{-1}, var_{-1})
    # EMStats initializes based on lookahead.
    # If lookahead=1 (default for simple case?), stats.avg[0] = x[0].
    # But the recursive formula assumes we have a previous state.
    # Let's start recovering from t=1, using state at t=0.

    mu_0 = stats.avg[0]
    var_0 = stats.var[0]

    # x[0] is scalar, mu_0 is (1,)
    # Let's make everything (1,)
    x_recovered = [x[0].view(1)]

    mu_curr = mu_0
    var_curr = var_0

    print(
        f"t=0: x={x[0]:.4f}, mu={mu_0.item():.4f}, var={var_0.item():.4f}, z={z[0].item():.4f}"
    )

    for t in range(1, T):
        z_t = z[t]

        x_t, mu_next, var_next = inverse_zscore_step(z_t, alpha_val, mu_curr, var_curr)

        x_recovered.append(x_t)

        # Verify against actual stats
        mu_actual = stats.avg[t]
        var_actual = stats.var[t]

        print(f"t={t}: z={z_t.item():.4f}")
        print(
            f"  Rec: x={x_t.item():.4f}, mu={mu_next.item():.4f}, var={var_next.item():.4f}"
        )
        print(
            f"  Act: x={x[t].item():.4f}, mu={mu_actual.item():.4f}, var={var_actual.item():.4f}"
        )

        if not torch.allclose(x_t, x[t], atol=1e-5):
            print(f"Mismatch at t={t}!")
            break

        mu_curr = mu_next
        var_curr = var_next

    x_rec_tensor = torch.stack(x_recovered).flatten()
    if torch.allclose(x_rec_tensor, x, atol=1e-5):
        print("SUCCESS: Recovered x matches original x!")
    else:
        print("FAILURE: Mismatch found.")


if __name__ == "__main__":
    verify()
