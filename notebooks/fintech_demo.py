# %%
import sys
from pathlib import Path
from datetime import date, timedelta, datetime, timezone
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Add src to path
# Assuming this file is in auralab/notebooks/
sys.path.append(str(Path(__file__).parent.parent / "src"))

from auralab.fintech.dataset import FintechDataset

# Configure logging to see download progress
logging.basicConfig(level=logging.INFO)

# %%
# Define parameters
symbols = ("BTCUSDT", "ETHUSDT")
markets = ("usdtm",)  # Binance Futures
start_date = date(2023, 1, 1)
end_date = date(2023, 2, 1)  # Full month of January
sample_rate = timedelta(minutes=1)
batch_size = 1024  # Larger batch size for efficient iteration if needed

print(f"Initializing FintechDataset...")
print(f"Symbols: {symbols}")
print(f"Time Range: {start_date} to {end_date}")

# %%
# Create Dataset
# This will download and cache data if not present.
# Note: First run might take time to download 1 month of data for 2 assets.
ds = FintechDataset(
    symbols=symbols,
    markets=markets,
    start_date=start_date,
    end_date=end_date,
    sample_rate=sample_rate,
    batch_size=batch_size,
)

print(f"Dataset ready.")
print(f"Total steps: {ds.num_steps}")
print(f"Total batches: {len(ds)}")

# %%
# Access data for plotting
# FintechDataset stores full tensors in ds.dataset [T, S, M]
# We can access them directly for visualization.

# Extract time axis
# We need to reconstruct timestamps.
# Start timestamp in ms
start_ts = int(
    datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp()
    * 1000
)
step_ms = int(sample_rate.total_seconds() * 1000)
times = [
    datetime.fromtimestamp((start_ts + i * step_ms) / 1000, tz=timezone.utc)
    for i in range(ds.num_steps)
]

# %%
# Plotting with Plotly (Dark Theme)

fig = make_subplots(
    rows=len(symbols),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=symbols,
)

for i, symbol in enumerate(symbols):
    # Get data for this symbol (Index i in S dimension)
    # Market index 0 (since we only have "usdtm")
    market_idx = 0

    # Extract OHLC from tensors
    # Tensors are [T, S, M]
    high = ds.dataset.high[:, i, market_idx].numpy()
    low = ds.dataset.low[:, i, market_idx].numpy()
    vwap = ds.dataset.vwap[:, i, market_idx].numpy()
    vol = ds.dataset.vol[:, i, market_idx].numpy()

    # Filter out NaNs (empty bins)
    mask = ~torch.isnan(torch.from_numpy(high))

    # Plot Candlestick-like (using OHLC or just Line if preferred)
    # Since we have High, Low, VWAP, let's plot High/Low range and VWAP.
    # Or we can treat VWAP as Close for a rough candlestick.

    # Let's plot VWAP as the main line, and a shaded area for High/Low?
    # Or just standard lines.

    # Plot VWAP
    fig.add_trace(
        go.Scatter(
            x=times,
            y=vwap,
            # x=[t for j, t in enumerate(times) if mask[j]],
            # y=vwap[mask],
            mode="lines",
            name=f"{symbol} VWAP",
            line=dict(color="#00F0FF"),  # Cyan
        ),
        row=i + 1,
        col=1,
    )

    # Plot High/Low bands (optional, might be too noisy for a whole month)
    # Let's just plot VWAP for clarity on a month view.

    # Add Volume on secondary axis? Or just price for now.

fig.update_layout(
    title="Crypto Assets - January 2023 (1m Sample Rate)",
    template="plotly_dark",
    height=800,
    showlegend=True,
)

fig.show()

# %%
# Save to HTML for viewing
fig.write_html("fintech_demo_plot.html")
print("Plot saved to fintech_demo_plot.html")
