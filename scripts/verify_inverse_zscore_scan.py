import torch
from neuralab.emstats import emstats


def inverse_zscore_parallel(z, alpha, mu_0, var_0):
    """
    Recover x from z using parallel scan operations.

    Mathematical derivation:
    1. Variance update: V_t = V_{t-1} * gamma_t
       where gamma_t = (1 - alpha) / (1 - alpha * z_t^2)
       -> sigma_t = sigma_{t-1} * sqrt(gamma_t)
       This is a cumulative product.

    2. Mean update: mu_t = mu_{t-1} + k_t * sigma_{t-1}
       where k_t = (alpha * z_t * sqrt(gamma_t)) / (1 - alpha)
       This is a cumulative sum of the terms (k_t * sigma_{t-1}).

    3. Recover x: x_t = mu_{t-1} + (z_t * sigma_t) / (1 - alpha)
    """
    T = z.shape[0]
    device = z.device

    # 1. Compute gamma and g (sqrt(gamma))
    # Note: z is (T, 1) or (T,)
    denom = 1.0 - alpha * z**2
    gamma = (1.0 - alpha) / denom
    g = torch.sqrt(gamma)  # (T,)

    # Compute sigma sequence
    # sigma_t = sigma_0 * prod(g_1...g_t)
    # We need to prepend 1.0 for cumprod to align correctly or handle indices
    # sigma_seq[t] corresponds to sigma_t
    # sigma_seq[-1] corresponds to sigma_0 (initial)

    # Let's use cumprod
    g_cumprod = torch.cumprod(g, dim=0)
    sigma_0 = torch.sqrt(var_0)
    sigma_seq = sigma_0 * g_cumprod

    # sigma_prev (sigma_{t-1}) is sigma_seq shifted: [sigma_0, sigma_1, ..., sigma_{T-1}]
    sigma_prev = torch.cat([sigma_0.view(1), sigma_seq[:-1]])

    # 2. Compute mu sequence
    # mu_t = mu_{t-1} + k_t * sigma_{t-1}
    # k_t = (alpha * z_t * g_t) / (1 - alpha)
    k = (alpha * z * g) / (1.0 - alpha)

    delta_mu = k * sigma_prev
    mu_cum_delta = torch.cumsum(delta_mu, dim=0)
    mu_seq = mu_0 + mu_cum_delta

    # mu_prev is mu_seq shifted
    mu_prev = torch.cat([mu_0.view(1), mu_seq[:-1]])

    # 3. Recover x
    # x_t = mu_{t-1} + (z_t * sigma_t) / (1 - alpha)
    x_rec = mu_prev + (z * sigma_seq) / (1.0 - alpha)

    return x_rec


def verify():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 100
    x = torch.randn(T, device=device)
    alpha_val = 0.1
    alpha = torch.tensor([alpha_val], device=device)

    # Forward pass to get z
    stats = emstats(x, alpha)
    z = stats.zscore.flatten()  # (T,)

    # Initial state
    mu_0 = stats.avg[0]
    var_0 = stats.var[0]

    # We start recovering from t=1
    # z_input should be z[1:]
    z_input = z[1:]
    x_target = x[1:]

    print("Running parallel inversion...")
    x_rec = inverse_zscore_parallel(z_input, alpha_val, mu_0, var_0)

    # Compare
    if torch.allclose(x_rec, x_target, atol=1e-5):
        print("SUCCESS: Parallel recovery matches original x!")
        print(f"Max error: {(x_rec - x_target).abs().max().item():.2e}")
    else:
        print("FAILURE: Mismatch found.")
        print(f"Max error: {(x_rec - x_target).abs().max().item():.2e}")
        # print first few
        for i in range(5):
            print(f"t={i+1}: rec={x_rec[i].item():.4f}, act={x_target[i].item():.4f}")


if __name__ == "__main__":
    verify()
