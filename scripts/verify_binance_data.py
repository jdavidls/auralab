import sys
from pathlib import Path
from datetime import date
import torch
import plotly.graph_objects as go
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from neuralab.fintech.binance import BinanceFetcher

logging.basicConfig(level=logging.INFO)


def verify():
    fetcher = BinanceFetcher(cache_dir=".cache/binance")
    symbol = "BTCUSDT"
    day = date(2023, 1, 1)

    print(f"Fetching {symbol} for {day}...")
    data = fetcher.fetch_day(symbol, day, "usdtm")

    print(f"Data fetched!")
    print(f"Time steps: {len(data.time)}")
    print(f"Price range: {data.price.min().item():.2f} - {data.price.max().item():.2f}")
    print(f"Total Volume: {data.qty.sum().item():.4f}")

    # Resample to 1-minute candles for plotting (simple OHLC)
    # This is a quick verification, not a full feature

    # Convert time to minutes from start of day
    start_ms = data.time[0].item()
    minutes = (data.time - start_ms) // 60000

    unique_mins, inverse = torch.unique(minutes, return_inverse=True)

    # Calculate OHLC
    # Open: first price in minute
    # High: max price
    # Low: min price
    # Close: last price

    # We can use scatter_reduce for min/max
    highs = torch.zeros_like(unique_mins, dtype=torch.float32).scatter_reduce_(
        0, inverse, data.price, reduce="amax", include_self=False
    )
    lows = torch.zeros_like(unique_mins, dtype=torch.float32).scatter_reduce_(
        0, inverse, data.price, reduce="amin", include_self=False
    )

    # For Open/Close, it's a bit trickier without stable sort or first/last reduction directly.
    # But since data is time-sorted:
    # Open is price at first index of each minute
    # Close is price at last index of each minute

    # Find first occurrence of each minute
    # unique_consecutive might be better but unique returns sorted unique elements

    # Let's just plot raw price for a subset to keep it simple and fast
    subset = 10000  # Plot first 10k trades

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.time[:subset].numpy(),
            y=data.price[:subset].numpy(),
            mode="lines",
            name="Price",
        )
    )

    fig.update_layout(title=f"{symbol} Trades (First {subset}) on {day}")
    fig.write_html("binance_verification.html")
    print("Plot saved to binance_verification.html")


if __name__ == "__main__":
    verify()
