import torch
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
from neuralab.portfolio import portfolio, signed_softmax


def run_comparison():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Common settings
    n_epochs = 1000
    lr = 0.01
    T, Assets = 100, 10

    # 1. Data Generation
    # Synthetic returns: mean 0.001, std 0.02
    returns = torch.randn(T, Assets, device=device) * 0.02 + 0.001

    # Portfolio Data: Log Prices
    log_prices = torch.cumsum(returns, dim=0)
    log_prices = torch.cat([torch.zeros(1, Assets, device=device), log_prices], dim=0)

    # Classification Data: Directional Targets (1 if ret > 0, else 0)
    # We predict direction for time t based on... well, we are optimizing 'weights' or 'signals' directly.
    # To make it comparable, we optimize a "signal" tensor for both.
    targets = (returns > 0).float()

    # 2. Initialize Parameters (Identical for both)
    # We will optimize 'logits' directly.
    # For Portfolio: weights = signed_softmax(logits)
    # For Classif:   probs   = sigmoid(logits)
    # Note: Portfolio weights are (T+1, Assets), but returns are (T, Assets).
    # To align, let's optimize logits of shape (T, Assets) corresponding to the active returns.
    # (ignoring the first/last alignment nuances for this rough magnitude check)

    logits_init = torch.randn(T, Assets, device=device) * 0.1

    # Model A: Portfolio Optimization
    logits_port = logits_init.clone().detach().requires_grad_(True)
    opt_port = optim.Adam([logits_port], lr=lr)
    grad_mse_port = []

    # Initialize loss history
    loss_hist_port = []
    loss_hist_class = []

    # Model B: Directional Classification
    logits_class = logits_init.clone().detach().requires_grad_(True)
    opt_class = optim.Adam([logits_class], lr=lr)
    loss_fn_bce = nn.BCELoss()
    grad_mse_class = []

    print(f"{'Epoch':<5} | {'Port MSE':<12} | {'Class MSE':<12}")
    print("-" * 40)

    for epoch in range(n_epochs):
        # --- Portfolio Step ---
        opt_port.zero_grad()
        # Pad logits to match portfolio expectation (T+1)
        # We'll just pad with zeros for the extra step to keep shapes simple
        logits_port_padded = torch.cat(
            [torch.zeros(1, Assets, device=device), logits_port], dim=0
        )

        w_port = signed_softmax(logits_port_padded, dim=-1)
        stats = portfolio(
            log_prices, w_port, cost_bps=0.01, a_dim=-1, sample_rate=timedelta(days=1)
        )
        loss_port = stats.max_perf_loss
        loss_port.backward()

        loss_hist_port.append(loss_port.item())
        if logits_port.grad is not None:
            grad_mse_port.append(logits_port.grad.pow(2).mean().item())
        opt_port.step()

        # --- Classification Step ---
        opt_class.zero_grad()
        probs = torch.sigmoid(logits_class)
        loss_class = loss_fn_bce(probs, targets)
        loss_class.backward()

        loss_hist_class.append(loss_class.item())
        if logits_class.grad is not None:
            grad_mse_class.append(logits_class.grad.pow(2).mean().item())
        opt_class.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"{epoch+1:<5} | {grad_mse_port[-1]:<12.2e} | {grad_mse_class[-1]:<12.2e}"
            )

    print("-" * 40)
    print("-" * 40)
    print("Gradient MSE Statistics (Min | Max | Avg):")

    port_tensor = torch.tensor(grad_mse_port)
    class_tensor = torch.tensor(grad_mse_class)

    print(
        f"Portfolio (Max Perf): {port_tensor.min():.4e} | {port_tensor.max():.4e} | {port_tensor.mean():.4e}"
    )
    print(
        f"Classification (BCE): {class_tensor.min():.4e} | {class_tensor.max():.4e} | {class_tensor.mean():.4e}"
    )

    # --- Plotting ---
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    pio.templates.default = "plotly_dark"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Gradient MSE (Log Scale)", "Loss History"),
    )

    # Row 1: Gradients
    fig.add_trace(
        go.Scatter(y=grad_mse_port, name="Grad MSE (Portfolio)", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=grad_mse_class, name="Grad MSE (Classif)", mode="lines"),
        row=1,
        col=1,
    )
    fig.update_yaxes(type="log", title_text="MSE", row=1, col=1)

    # Row 2: Losses (Dual Y-axis might be needed if scales differ vastly, but let's try single first)
    # Portfolio loss is negative return (approx -0.02 to -0.05 maybe?), BCE is 0.69 down to 0.
    # They are on different scales. Let's use secondary y-axis for Classification Loss.

    fig.add_trace(
        go.Scatter(
            y=loss_hist_port,
            name="Loss (Portfolio)",
            mode="lines",
            line=dict(color="cyan"),
        ),
        row=2,
        col=1,
    )
    # Create a secondary y-axis trace for BCE is tricky with make_subplots rows/cols directly without specs.
    # Simpler: Just plot them. If scales are wild, we'll see.
    fig.add_trace(
        go.Scatter(
            y=loss_hist_class,
            name="Loss (Classif)",
            mode="lines",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Loss Value", row=2, col=1)

    fig.update_layout(
        height=800, title_text="Gradient & Loss Comparison: Portfolio vs Classification"
    )
    fig.show()


if __name__ == "__main__":
    run_comparison()
