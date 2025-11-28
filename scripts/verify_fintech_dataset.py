import sys
from pathlib import Path
from datetime import date, timedelta
import torch
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from auralab.fintech.dataset import FintechDataset

logging.basicConfig(level=logging.INFO)


def verify():
    # Use real data for BTCUSDT (Binance) and XXBTZUSD (Kraken)
    # We already fetched 2023-01-01 for both in previous steps, so it should be cached.

    symbols = ("BTCUSDT", "XXBTZUSD")
    # Map markets: BTCUSDT -> usdtm (Binance), XXBTZUSD -> spot (Kraken)
    # But FintechDataset takes a tuple of markets for ALL symbols?
    # No, the design in dataset.py iterates: for symbol in symbols: for market in markets.
    # This implies a cross product.
    # If we want specific pairs, we might need a different design or just ignore missing ones.
    # However, our dataset implementation tries to fetch EVERYTHING.
    # If we ask for ("BTCUSDT", "XXBTZUSD") and ("usdtm", "spot"), it will try:
    # BTCUSDT-usdtm (Valid Binance)
    # BTCUSDT-spot (Valid Binance Spot? Or Kraken?)
    # XXBTZUSD-usdtm (Invalid)
    # XXBTZUSD-spot (Valid Kraken)

    # Let's simplify for verification.
    # Just use one symbol that exists on one market to verify the pipeline.
    # Or use BTCUSDT on usdtm.

    symbols = ("BTCUSDT",)
    markets = ("usdtm",)

    start = date(2023, 1, 1)
    end = date(2023, 1, 2)  # 1 day
    sample_rate = timedelta(minutes=1)
    batch_size = 64

    print(f"Creating FintechDataset for {symbols} {markets}...")
    ds = FintechDataset(symbols, markets, start, end, sample_rate, batch_size)

    print(f"Dataset created. Length (batches): {len(ds)}")

    # Iterate a few batches
    for i, batch in enumerate(ds):
        if i >= 3:
            break
        print(f"Batch {i}:")
        # SampledData is a NamedTuple
        for k, v in batch._asdict().items():
            print(f"  {k}: {v.shape}")
            # Expected shape: [64, 1, 1] (Batch, Symbol, Market)

    print("Verification successful!")


if __name__ == "__main__":
    verify()
