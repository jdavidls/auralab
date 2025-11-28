import sys
from pathlib import Path
from datetime import date, datetime, timezone
import torch
import plotly.graph_objects as go
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from auralab.fintech.binance import BinanceFetcher
from auralab.fintech.sampling import sample_trades

logging.basicConfig(level=logging.INFO)


def verify():
    fetcher = BinanceFetcher(cache_dir=".cache/binance")
    symbol = "BTCUSDT"
    day = date(2023, 1, 1)

    print(f"Fetching {symbol} for {day}...")
    trades = fetcher.fetch_day(symbol, day, "usdtm")

    # Resample to 1 minute candles
    start_ms = trades.time[0].item()
    # Align start to minute boundary
    start_ms = (start_ms // 60000) * 60000
    end_ms = trades.time[-1].item()
    step_ms = 60000  # 1 minute

    print(f"Resampling to 1 minute candles...")
    sampled = sample_trades(trades, step_ms, start_ms, end_ms)

    print(f"Sampled {len(sampled.high)} candles.")

    # Plot OHLC
    # Filter out NaNs for plotting (empty bins)
    mask = ~torch.isnan(sampled.high)

    # Create time axis
    times = [
        datetime.fromtimestamp((start_ms + i * step_ms) / 1000, tz=timezone.utc)
        for i in range(len(sampled.high))
    ]
    times = [t for i, t in enumerate(times) if mask[i]]

    open_p = sampled.vwap[mask]  # Approximation, usually Open is first price.
    # Our sample_trades doesn't return Open/Close yet, only High/Low/VWAP.
    # Let's use VWAP as Open/Close for visualization or just plot High/Low/VWAP lines.

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times,
            y=sampled.high[mask].numpy(),
            mode="lines",
            name="High",
            line=dict(color="green", width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=sampled.low[mask].numpy(),
            mode="lines",
            name="Low",
            line=dict(color="red", width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=sampled.vwap[mask].numpy(),
            mode="lines",
            name="VWAP",
            line=dict(color="blue", width=2),
        )
    )

    fig.update_layout(title=f"{symbol} 1m Resampled Data (High/Low/VWAP)")
    fig.write_html("sampling_verification.html")
    print("Plot saved to sampling_verification.html")


if __name__ == "__main__":
    verify()
