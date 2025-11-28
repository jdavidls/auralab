# %%
import torch
import torch.optim as optim
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from auralab.portfolio import portfolio, signed_softmax

# Set dark theme
pio.templates.default = "plotly_dark"

# 1. Generate synthetic data
torch.manual_seed(42)
T, Assets = 100, 3
# Random returns: mean 0.001, std 0.02
returns = torch.randn(T, Assets) * 0.02 + 0.001
# Prices start at 100
log_prices = torch.cumsum(returns, dim=0)
log_prices = torch.cat([torch.zeros(1, Assets), log_prices], dim=0) + 4.605

# 2. Initialize weights for two models
# We want to compare optimizing for Total Return vs Sharpe Ratio
# Initialize with same random values
# Weights must match log_prices length (T+1)
weights_init = torch.randn(T + 1, Assets) * 0.1

# Model 1: Optimize for Total Return
weights_return = weights_init.clone().detach().requires_grad_(True)
opt_return = optim.Adam([weights_return], lr=0.01)

# Model 2: Optimize for Sharpe Ratio
weights_sharpe = weights_init.clone().detach().requires_grad_(True)
opt_sharpe = optim.Adam([weights_sharpe], lr=0.01)

# 3. Optimization loop
n_epochs = 4000
mode = "cross"  # Use cross mode for competition

print(f"Starting comparative optimization for {n_epochs} epochs in '{mode}' mode...")


grad_mse_ret = []
grad_mse_shp = []

for epoch in range(n_epochs):
    opt_return.zero_grad()
    opt_sharpe.zero_grad()

    # Constrain weights
    if mode == "isolated":
        w_ret = torch.tanh(weights_return)
        w_shp = torch.tanh(weights_sharpe)
    elif mode == "cross":
        w_ret = signed_softmax(weights_return, dim=-1)
        w_shp = signed_softmax(weights_sharpe, dim=-1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Calculate stats
    # Assume hourly data
    stats_ret = portfolio(
        log_prices, w_ret, cost_bps=0.01, a_dim=-1, sample_rate=timedelta(hours=1)
    )
    stats_shp = portfolio(
        log_prices, w_shp, cost_bps=0.01, a_dim=-1, sample_rate=timedelta(hours=1)
    )

    # Compute loss
    # Return Model: Minimize negative annualized performance
    loss_ret = stats_ret.max_perf_loss

    # Sharpe Model: Minimize negative portfolio Sharpe
    loss_shp = stats_shp.cross_loss_sharpe

    # Backward pass
    opt_return.zero_grad()
    loss_ret.backward()
    opt_return.step()

    opt_sharpe.zero_grad()
    loss_shp.backward()
    opt_sharpe.step()

    # Log gradient MSE
    if weights_return.grad is not None:
        grad_mse_ret.append(weights_return.grad.pow(2).mean().item())
    if weights_sharpe.grad is not None:
        grad_mse_shp.append(weights_sharpe.grad.pow(2).mean().item())

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"Return Model: Ret={stats_ret.iso_total_perf.sum().item():.4f}, Sharpe={stats_ret.iso_sharpe.mean().item():.4f} | "
            f"Sharpe Model: Ret={stats_shp.iso_total_perf.sum().item():.4f}, Sharpe={stats_shp.iso_sharpe.mean().item():.4f}"
        )

# %%
# Final results
if mode == "isolated":
    final_w_ret = torch.tanh(weights_return)
    final_w_shp = torch.tanh(weights_sharpe)
elif mode == "cross":
    final_w_ret = signed_softmax(weights_return, dim=-1)
    final_w_shp = signed_softmax(weights_sharpe, dim=-1)
else:
    raise ValueError(f"Unknown mode: {mode}")

stats_ret = portfolio(log_prices, final_w_ret, cost_bps=0.01, a_dim=-1)
stats_shp = portfolio(log_prices, final_w_shp, cost_bps=0.01, a_dim=-1)

print("\nOptimization complete.")
print("-" * 60)
print(f"{'Metric':<20} | {'Return Model':<15} | {'Sharpe Model':<15}")
print("-" * 60)
print(
    f"{'Total Return (Sum)':<20} | {stats_ret.iso_total_perf.sum().item():<15.4f} | {stats_shp.iso_total_perf.sum().item():<15.4f}"
)
print(
    f"{'Sharpe Ratio (Mean)':<20} | {stats_ret.iso_sharpe.mean().item():<15.4f} | {stats_shp.iso_sharpe.mean().item():<15.4f}"
)
print(
    f"{'Portfolio Sharpe':<20} | {stats_ret.cross_sharpe.item():<15.4f} | {stats_shp.cross_sharpe.item():<15.4f}"
)
print("-" * 60)
print(f"Final Gradient MSE (Return Model): {grad_mse_ret[-1]:.6e}")
print(f"Final Gradient MSE (Sharpe Model): {grad_mse_shp[-1]:.6e}")
print("-" * 60)

# %%
# Visualization
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        "Asset Prices",
        "Weights (Return Model)",
        "Weights (Sharpe Model)",
        "Gradient MSE (Log Scale)",
    ),
)

# 1. Asset Prices
for i in range(Assets):
    fig.add_trace(
        go.Scatter(y=torch.exp(log_prices[:, i]).numpy(), name=f"Asset {i}"),
        row=1,
        col=1,
    )

# 2. Portfolio Weights - Return Model
fig.add_trace(
    go.Heatmap(
        z=final_w_ret.detach().T.numpy(),
        x=list(range(T)),
        y=[f"Asset {i}" for i in range(Assets)],
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Weight", len=0.25, y=0.5),
        showscale=False,  # Share colorbar or hide to avoid clutter? Let's keep one or hide.
    ),
    row=2,
    col=1,
)

# 3. Portfolio Weights - Sharpe Model
fig.add_trace(
    go.Heatmap(
        z=final_w_shp.detach().T.numpy(),
        x=list(range(T)),
        y=[f"Asset {i}" for i in range(Assets)],
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Weight", len=0.25, y=0.15),
    ),
    row=3,
    col=1,
)

# 4. Gradient MSE
fig.add_trace(
    go.Scatter(y=grad_mse_ret, name="Grad MSE (Return)", mode="lines"),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(y=grad_mse_shp, name="Grad MSE (Sharpe)", mode="lines"),
    row=4,
    col=1,
)
fig.update_yaxes(type="log", row=4, col=1)

fig.update_layout(
    height=900, title_text="Comparative Optimization Results", showlegend=True
)
fig.show()
