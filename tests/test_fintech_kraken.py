import unittest
from unittest.mock import patch, MagicMock
import torch
from datetime import date, datetime, timezone
from pathlib import Path
import sys
import json
import io

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from auralab.fintech.sources.kraken import KrakenFetcher
from auralab.fintech.sources.core import TradeData, TradingPair, Market


class TestKrakenFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = KrakenFetcher(
            cache_dir="/tmp/test_cache_kraken", rate_limit_delay=0.0
        )

    @patch("auralab.fintech.sources.kraken.urllib.request.urlopen")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    @patch("torch.save")
    def test_fetch_day_pagination(
        self, mock_save, mock_mkdir, mock_exists, mock_urlopen
    ):
        # Setup mocks
        mock_exists.return_value = False

        # Mock API responses
        # We need to simulate pagination.
        # Day: 2023-01-01. Start TS: 1672531200. End TS: 1672617600.

        # Response 1: Trades before end of day, last pointer still within day
        # Response 2: Trades crossing end of day, last pointer after day

        start_ts = 1672531200
        mid_ts = 1672574400
        end_ts = 1672617600

        # Trade format: [price, volume, time, buy/sell, market/limit, misc, id]

        resp1_data = {
            "error": [],
            "result": {
                "XXBTZUSD": [
                    ["16500.0", "0.1", str(start_ts + 10), "b", "l", "", 1],
                    ["16501.0", "0.2", str(mid_ts - 10), "s", "m", "", 2],
                ],
                "last": str(mid_ts * 1_000_000_000),  # last is ns
            },
        }

        resp2_data = {
            "error": [],
            "result": {
                "XXBTZUSD": [
                    ["16502.0", "0.3", str(mid_ts + 10), "b", "l", "", 3],
                    [
                        "16503.0",
                        "0.4",
                        str(end_ts + 10),
                        "s",
                        "m",
                        "",
                        4,
                    ],  # This one is outside
                ],
                "last": str((end_ts + 100) * 1_000_000_000),
            },
        }

        # Mock urlopen context manager
        mock_response1 = MagicMock()
        mock_response1.read.return_value = json.dumps(resp1_data).encode("utf-8")

        mock_response2 = MagicMock()
        mock_response2.read.return_value = json.dumps(resp2_data).encode("utf-8")

        # urlopen returns a context manager
        cm1 = MagicMock()
        cm1.__enter__.return_value = mock_response1
        cm1.__exit__.return_value = None

        cm2 = MagicMock()
        cm2.__enter__.return_value = mock_response2
        cm2.__exit__.return_value = None

        mock_urlopen.side_effect = [cm1, cm2]

        # Execute
        day = date(2023, 1, 1)
        pair = TradingPair("BTC", "USD")
        market = Market(Market.Platform.KRAKEN, "spot")
        data = self.fetcher.fetch_day(pair, day, market)

        # Verify
        self.assertEqual(mock_urlopen.call_count, 2)

        # Check data
        # Should have 3 trades (the 4th one is outside the day)
        self.assertEqual(len(data.time), 3)

        # Verify timestamps (ms)
        self.assertEqual(data.time[0].item(), (start_ts + 10) * 1000)
        self.assertEqual(data.time[1].item(), (mid_ts - 10) * 1000)
        self.assertEqual(data.time[2].item(), (mid_ts + 10) * 1000)

        # Verify is_buyer_maker
        # 'b' -> taker buy -> maker sell -> False
        # 's' -> taker sell -> maker buy -> True
        self.assertFalse(data.is_buyer_maker[0].item())
        self.assertTrue(data.is_buyer_maker[1].item())
        self.assertFalse(data.is_buyer_maker[2].item())

    @patch("pathlib.Path.exists")
    @patch("torch.load")
    def test_fetch_day_cached(self, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_data = TradeData(
            time=torch.tensor([1]),
            price=torch.tensor([1.0]),
            qty=torch.tensor([1.0]),
            is_buyer_maker=torch.tensor([True]),
        )
        mock_load.return_value = mock_data

        day = date(2023, 1, 1)
        pair = TradingPair("BTC", "USD")
        market = Market(Market.Platform.KRAKEN, "spot")
        data = self.fetcher.fetch_day(pair, day, market)

        self.assertEqual(data, mock_data)
        mock_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
