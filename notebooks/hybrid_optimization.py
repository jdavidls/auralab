# %%
import torch
import torch.optim as optim
import plotly.io as pio
import plotly.graph_objects as go
from datetime import timedelta
from auralab.portfolio import portfolio, signed_softmax

# Set dark theme
pio.templates.default = "plotly_dark"

# 1. Generate synthetic data
torch.manual_seed(42)
T, Assets = 100, 5
# Random returns: mean 0.001, std 0.02
returns = torch.randn(T, Assets) * 0.02 + 0.001
# Prices start at 100
log_prices = torch.cumsum(returns, dim=0)
log_prices = torch.cat([torch.zeros(1, Assets), log_prices], dim=0) + 4.605

# 2. Define Scenarios
scenarios = {
    "Baseline (Max Return)": {"return_target": None, "sharpe_target": None},
    "Dynamic (Ret>10%, Shp>2)": {"return_target": 0.10, "sharpe_target": 2.0},
    "High Ret Target (>50%)": {"return_target": 0.50, "sharpe_target": 2.0},
    "High Sharpe Target (>5)": {"return_target": 0.10, "sharpe_target": 5.0},
}

results = {}
n_epochs = 2000

print(f"Starting Hybrid Optimization (Dynamic) for {n_epochs} epochs...")
print(f"{'Scenario':<30} | {'Return':<10} | {'Sharpe':<10}")
print("-" * 55)

for name, params in scenarios.items():
    # Initialize weights
    weights = torch.randn(T + 1, Assets, requires_grad=True)
    optimizer = optim.Adam([weights], lr=0.01)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Use cross mode (signed_softmax)
        w = signed_softmax(weights, dim=-1)

        stats = portfolio(
            log_prices, w, cost_bps=0.01, a_dim=-1, sample_rate=timedelta(days=1)
        )

        # Calculate Hybrid Loss
        loss = stats.hybrid_loss(
            return_target=params["return_target"],
            sharpe_target=params["sharpe_target"],
            penalty_weight=100.0,
        )

        loss.backward()
        optimizer.step()

    # Final Stats
    with torch.no_grad():
        w_final = signed_softmax(weights, dim=-1)
        stats_final = portfolio(
            log_prices, w_final, cost_bps=0.01, a_dim=-1, sample_rate=timedelta(days=1)
        )

        # Calculate Portfolio Return (Annualized)
        r = stats_final.cross_net_perf.narrow(
            stats_final.t_dim, 0, stats_final.cross_net_perf.size(stats_final.t_dim) - 1
        )
        final_ret = (r.mean() * stats_final.steps_per_year).item()
        final_shp = stats_final.cross_sharpe.item()

        results[name] = {"Return": final_ret, "Sharpe": final_shp}
        print(f"{name:<25} | {final_ret:<10.4f} | {final_shp:<10.4f}")

print("-" * 50)

# %%
# Visualization
fig = go.Figure()

for name, res in results.items():
    fig.add_trace(
        go.Scatter(
            x=[res["Sharpe"]],
            y=[res["Return"]],
            mode="markers+text",
            name=name,
            text=[name],
            textposition="top center",
            marker=dict(size=15),
        )
    )

fig.update_layout(
    title="Hybrid Optimization Results: Return vs Sharpe",
    xaxis_title="Portfolio Sharpe Ratio",
    yaxis_title="Annualized Return",
    height=600,
    width=800,
)
fig.show()
