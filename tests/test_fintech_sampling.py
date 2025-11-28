import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from auralab.fintech.sources.core import TradeData
from auralab.fintech.sampling import sample_trades


class TestSampling(unittest.TestCase):
    def test_sample_trades(self):
        # Create synthetic data
        # 3 trades in 2 bins.
        # Bin 0: 2 trades
        # Bin 1: 1 trade

        start_ms = 1000
        step_ms = 1000
        end_ms = 3000  # 2 bins: [1000, 2000), [2000, 3000)

        times = torch.tensor([1100, 1500, 2100], dtype=torch.int64)
        prices = torch.tensor([100.0, 102.0, 105.0], dtype=torch.float32)
        qtys = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)
        # is_buyer_maker: True (Sell), False (Buy), True (Sell)
        is_buyer_maker = torch.tensor([True, False, True], dtype=torch.bool)

        trades = TradeData(times, prices, qtys, is_buyer_maker)

        sampled = sample_trades(trades, step_ms, start_ms, end_ms)

        # Verify Bin 0
        # Trades: (100, 1.0, Sell), (102, 2.0, Buy)
        # Vol: 1.0 + 2.0 = 3.0
        # VWAP: (100*1 + 102*2) / 3 = (100 + 204) / 3 = 304 / 3 = 101.333
        # High: 102.0
        # Low: 100.0
        # Bid Vol (Buy): 2.0
        # Ask Vol (Sell): 1.0

        self.assertAlmostEqual(sampled.vol[0].item(), 3.0)
        self.assertAlmostEqual(sampled.vwap[0].item(), 101.333333, places=4)
        self.assertAlmostEqual(sampled.high[0].item(), 102.0)
        self.assertAlmostEqual(sampled.low[0].item(), 100.0)
        self.assertAlmostEqual(sampled.bid_vol[0].item(), 2.0)
        self.assertAlmostEqual(sampled.ask_vol[0].item(), 1.0)

        # Verify Bin 1
        # Trades: (105, 5.0, Sell)
        # Vol: 5.0
        # VWAP: 105.0
        # High: 105.0
        # Low: 105.0
        # Bid Vol: 0.0
        # Ask Vol: 5.0

        self.assertAlmostEqual(sampled.vol[1].item(), 5.0)
        self.assertAlmostEqual(sampled.vwap[1].item(), 105.0)
        self.assertAlmostEqual(sampled.high[1].item(), 105.0)
        self.assertAlmostEqual(sampled.low[1].item(), 105.0)
        self.assertAlmostEqual(sampled.bid_vol[1].item(), 0.0)
        self.assertAlmostEqual(sampled.ask_vol[1].item(), 5.0)

    def test_empty_bin(self):
        start_ms = 0
        step_ms = 1000
        end_ms = 2000

        # Trade only in bin 0
        times = torch.tensor([500], dtype=torch.int64)
        prices = torch.tensor([100.0], dtype=torch.float32)
        qtys = torch.tensor([1.0], dtype=torch.float32)
        is_buyer_maker = torch.tensor([False], dtype=torch.bool)

        trades = TradeData(times, prices, qtys, is_buyer_maker)

        sampled = sample_trades(trades, step_ms, start_ms, end_ms)

        # Bin 1 should be empty
        # Bin 1 should be empty but forward filled from Bin 0
        self.assertEqual(sampled.vol[1].item(), 0.0)
        # Forward filled prices
        self.assertEqual(sampled.high[1].item(), 100.0)
        self.assertEqual(sampled.low[1].item(), 100.0)
        self.assertEqual(sampled.vwap[1].item(), 100.0)


if __name__ == "__main__":
    unittest.main()
