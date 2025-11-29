import unittest
import os
from unittest.mock import patch
from datetime import date, timedelta
import importlib
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import auralab.fintech.dataset
from auralab.fintech.dataset import FintechDataset


class TestFintechDefaults(unittest.TestCase):
    def test_defaults(self):
        # Reload to ensure defaults (though they are class attributes evaluated at import time)
        importlib.reload(auralab.fintech.dataset)
        from auralab.fintech.dataset import FintechDataset

        self.assertEqual(
            FintechDataset.DEFAULT_SYMBOLS, ("BTC-USDT", "ETH-USDT", "ETH-BTC")
        )
        self.assertEqual(FintechDataset.DEFAULT_MARKETS, ("binance-usdtm",))
        self.assertEqual(FintechDataset.DEFAULT_START_DATE, date(2023, 1, 1))
        self.assertEqual(FintechDataset.DEFAULT_END_DATE, date(2023, 2, 1))
        self.assertEqual(FintechDataset.DEFAULT_SAMPLE_RATE, timedelta(minutes=1))
        self.assertEqual(FintechDataset.DEFAULT_BATCH_SIZE, 64)

    @patch.dict(
        os.environ,
        {
            "FINTECH_SYMBOLS": "SOL-USDT, ADA-USDT",
            "FINTECH_MARKETS": "kraken-spot",
            "FINTECH_START_DATE": "2024-01-01",
            "FINTECH_END_DATE": "2024-01-02",
            "FINTECH_SAMPLE_RATE_MIN": "60",
            "FINTECH_BATCH_SIZE": "128",
        },
    )
    def test_env_vars(self):
        # We need to reload the module to pick up env vars because they are evaluated at module level
        importlib.reload(auralab.fintech.dataset)
        from auralab.fintech.dataset import FintechDataset

        self.assertEqual(FintechDataset.DEFAULT_SYMBOLS, ("SOL-USDT", "ADA-USDT"))
        self.assertEqual(FintechDataset.DEFAULT_MARKETS, ("kraken-spot",))
        self.assertEqual(FintechDataset.DEFAULT_START_DATE, date(2024, 1, 1))
        self.assertEqual(FintechDataset.DEFAULT_END_DATE, date(2024, 1, 2))
        self.assertEqual(FintechDataset.DEFAULT_SAMPLE_RATE, timedelta(minutes=60))
        self.assertEqual(FintechDataset.DEFAULT_BATCH_SIZE, 128)


if __name__ == "__main__":
    unittest.main()
