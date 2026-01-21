import torch
from neuralab.ema import ema
from neuralab.emstats import emstats


def verify_lookahead():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Create a simple tensor: [1, 2, 3, 4, 5]
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    alpha = torch.tensor([0.5], device=device)

    print("\n--- Verifying EMA lookahead ---")

    # Case 1: lookahead=1 (default behavior)
    # Initial state should be x[0] = 1.0
    y1 = ema(x, alpha, lookahead=1)
    print(f"EMA lookahead=1, first element: {y1[0].item()}")
    assert torch.isclose(
        y1[0], torch.tensor(1.0, device=device)
    ), "EMA lookahead=1 failed"

    # Case 2: lookahead=3
    # Initial state S0 = mean(1, 2, 3) = 2.0
    # y0 = alpha * x0 + (1 - alpha) * S0
    # y0 = 0.5 * 1.0 + 0.5 * 2.0 = 1.5
    y3 = ema(x, alpha, lookahead=3)
    print(f"EMA lookahead=3, first element: {y3[0].item()}")
    assert torch.isclose(
        y3[0], torch.tensor(1.5, device=device)
    ), "EMA lookahead=3 failed"

    print("\n--- Verifying EMStats init_window ---")

    # Case 3: EMStats with init_window=3
    stats = emstats(x, alpha, dim=0, init_window=3)
    avg = stats.avg
    print(f"EMStats init_window=3, first element: {avg[0].item()}")
    assert torch.isclose(
        avg[0], torch.tensor(1.5, device=device)
    ), "EMStats init_window=3 failed"

    print("\nVerification successful!")


if __name__ == "__main__":
    verify_lookahead()
