# %%
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from auralab.ema import ema_decay
from auralab.fintech.dataset import FintechDataset
from auralab.emstats import emstats

# Initialize dataset
# Initialize dataset
ds = FintechDataset.default()

# Get VWAP data: [T, S, M]
# Note: ds.dataset is the SampledData named tuple containing the tensors
vwap = ds.dataset.vwap
symbols = ds.pairs
markets = ds.markets

# Ensure we have at least 2 symbols for covariance
if len(symbols) < 2:
    raise ValueError("Need at least 2 symbols for covariance plotting")

# Use first market for simplicity (e.g. usdtm)
market_idx = 0
market_name = markets[market_idx]

# Prepare data for first two symbols
s1_idx, s2_idx = 0, 1
sym1, sym2 = symbols[s1_idx], symbols[s2_idx]

price1 = vwap[:, s1_idx, market_idx]
price2 = vwap[:, s2_idx, market_idx]

# EMStats parameters
alpha = ema_decay(60)  # 60 minutos
# Ensure tensors are on the correct device for emstats if needed,
# but dataset loads to CPU by default. emstats might require CUDA if available?
# The demo used CUDA. Let's try to use CUDA if available for emstats, then back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

price1_dev = price1.to(device)
price2_dev = price2.to(device)
alpha_dev = torch.tensor(alpha, device=device)

# Compute stats for individual assets
stats1 = emstats(price1_dev, alpha_dev)
stats2 = emstats(price2_dev, alpha_dev)

ema1 = stats1.avg.cpu()
std1 = stats1.std.cpu()

ema2 = stats2.avg.cpu()
std2 = stats2.std.cpu()

z1 = stats1.zscore.cpu()
z2 = stats2.zscore.cpu()

# Compute Correlation
# Stack prices: [T, 2]
prices_stacked = torch.stack([price1_dev, price2_dev], dim=1)
stats_combined = emstats(prices_stacked, alpha_dev, dim=0)

# Correlation matrix: [T, 2, 2] (if full) or flattened upper triangular?
# Demo says: corr_matrix = stats_xy.corr(dim_c=1) returns flattened strictly upper triangular
corr_matrix = stats_combined.corr(dim_c=1)
# For 2 variables, strictly upper triangular is just 1 element: Corr(X, Y)
corr_12 = corr_matrix[:, 0].cpu()

# Create subplots
# Create subplots
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    specs=[[{"secondary_y": True}], [{}], [{}]],
    subplot_titles=("Asset Prices with EMA & StdDev", "Correlation", "Z-Score"),
)

# --- Row 1: Prices & EMStats ---

# Asset 1 (Left Axis)
fig.add_trace(
    go.Scatter(
        y=price1, name=f"{sym1} Price", line=dict(color="blue", width=1), opacity=0.5
    ),
    row=1,
    col=1,
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(y=ema1, name=f"{sym1} EMA", line=dict(color="blue", width=2)),
    row=1,
    col=1,
    secondary_y=False,
)

# Asset 1 Bands
fig.add_trace(
    go.Scatter(
        y=ema1 + 2 * std1,
        name=f"{sym1} +2std",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ),
    row=1,
    col=1,
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        y=ema1 - 2 * std1,
        name=f"{sym1} -2std",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0, 0, 255, 0.1)",
        showlegend=False,
        hoverinfo="skip",
    ),
    row=1,
    col=1,
    secondary_y=False,
)


# Asset 2 (Right Axis)
fig.add_trace(
    go.Scatter(
        y=price2, name=f"{sym2} Price", line=dict(color="red", width=1), opacity=0.5
    ),
    row=1,
    col=1,
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(y=ema2, name=f"{sym2} EMA", line=dict(color="red", width=2)),
    row=1,
    col=1,
    secondary_y=True,
)

# Asset 2 Bands
fig.add_trace(
    go.Scatter(
        y=ema2 + 2 * std2,
        name=f"{sym2} +2std",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ),
    row=1,
    col=1,
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(
        y=ema2 - 2 * std2,
        name=f"{sym2} -2std",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255, 0, 0, 0.1)",
        showlegend=False,
        hoverinfo="skip",
    ),
    row=1,
    col=1,
    secondary_y=True,
)


# --- Row 2: Correlation ---

fig.add_trace(
    go.Scatter(y=corr_12, name=f"Corr({sym1}, {sym2})", line=dict(color="purple")),
    row=2,
    col=1,
)


# --- Row 3: Z-Score ---

fig.add_trace(
    go.Scatter(y=z1, name=f"{sym1} Z-Score", line=dict(color="blue", width=1)),
    row=3,
    col=1,
)

fig.add_trace(
    go.Scatter(y=z2, name=f"{sym2} Z-Score", line=dict(color="red", width=1)),
    row=3,
    col=1,
)

# Add +/- 2 threshold lines for Z-Score
fig.add_hline(y=2, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=-2, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=0, line_color="black", row=3, col=1)


# Layout updates
fig.update_layout(
    title_text=f"Market Analysis: {sym1} vs {sym2} ({market_name})", height=1000
)

fig.update_yaxes(title_text=f"{sym1} Price", secondary_y=False, row=1, col=1)
fig.update_yaxes(title_text=f"{sym2} Price", secondary_y=True, row=1, col=1)
fig.update_yaxes(title_text="Correlation", row=2, col=1)
fig.update_yaxes(title_text="Z-Score", row=3, col=1)

fig.show()
