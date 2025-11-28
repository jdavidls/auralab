import sys
from pathlib import Path
from datetime import date
import torch
import plotly.graph_objects as go
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from auralab.fintech.kraken import KrakenFetcher

logging.basicConfig(level=logging.INFO)


def verify():
    fetcher = KrakenFetcher(cache_dir=".cache/kraken")
    symbol = "XXBTZUSD"  # Kraken BTC/USD pair name
    day = date(2023, 1, 1)

    print(f"Fetching {symbol} for {day}...")
    # This might take a while due to pagination and rate limits
    data = fetcher.fetch_day(symbol, day, "spot")

    print(f"Data fetched!")
    print(f"Time steps: {len(data.time)}")
    print(f"Price range: {data.price.min().item():.2f} - {data.price.max().item():.2f}")
    print(f"Total Volume: {data.qty.sum().item():.4f}")

    # Plot first 10k trades
    subset = 10000

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
    fig.write_html("kraken_verification.html")
    print("Plot saved to kraken_verification.html")


if __name__ == "__main__":
    verify()
