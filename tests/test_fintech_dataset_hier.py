import unittest
from unittest.mock import MagicMock, patch
import torch
from datetime import date, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from auralab.fintech.dataset import FintechDataset, SamplingDataset, TradingDataset
from auralab.fintech.sources.core import TradeData, SampledData


class TestHierarchicalDataset(unittest.TestCase):

    @patch("auralab.fintech.dataset.TradingDataset.ensure")
    def test_sampling_dataset(self, mock_ensure):
        # Mock raw trade data
        # 2023-01-01 00:00:00 UTC = 1672531200000 ms
        base_ts = 1672531200000
        mock_trades = TradeData(
            time=torch.tensor([base_ts + 1000, base_ts + 2000], dtype=torch.int64),
            price=torch.tensor([100.0, 101.0], dtype=torch.float32),
            qty=torch.tensor([1.0, 1.0], dtype=torch.float32),
            is_buyer_maker=torch.tensor([False, True], dtype=torch.bool),
        )
        mock_ensure.return_value = mock_trades

        # Create dataset
        start = date(2023, 1, 1)
        end = date(2023, 1, 2)
        ds = SamplingDataset(
            "BTCUSDT", "usdtm", start, end, 60000, cache_dir="/tmp/test_hier"
        )

        # Ensure (triggers sampling)
        with patch("auralab.fintech.dataset.torch.save") as mock_save:
            with patch("auralab.fintech.dataset.torch.load") as mock_load:
                with patch("pathlib.Path.exists") as mock_exists:
                    mock_exists.return_value = False  # Force compute

                    sampled = ds.ensure()

                    self.assertIsInstance(sampled, SampledData)
                    mock_save.assert_called()

    @patch("auralab.fintech.dataset.SamplingDataset.ensure")
    def test_fintech_dataset(self, mock_ensure):
        # Mock sampled data (T=10)
        T = 10
        mock_sampled = SampledData(
            high=torch.randn(T),
            low=torch.randn(T),
            vwap=torch.randn(T),
            vol=torch.randn(T),
            bid_vol=torch.randn(T),
            ask_vol=torch.randn(T),
        )
        mock_ensure.return_value = mock_sampled

        symbols = ("BTC", "ETH")
        markets = ("spot", "usdtm")
        start = date(2023, 1, 1)
        end = date(2023, 1, 2)
        sample_rate = timedelta(minutes=1)
        batch_size = 2

        ds = FintechDataset(symbols, markets, start, end, sample_rate, batch_size)

        # Check length
        # T=10, Batch=2 -> Len=5
        self.assertEqual(len(ds), 5)

        # Check item shape
        batch = ds[0]
        self.assertIsInstance(batch, SampledData)
        # Shape: [Batch, S, M] -> [2, 2, 2]
        self.assertEqual(batch.high.shape, (2, 2, 2))
        self.assertEqual(batch.vol.shape, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
