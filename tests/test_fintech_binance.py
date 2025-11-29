import unittest
from unittest.mock import patch, MagicMock, mock_open
import torch
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import io
import sys

# Add src to path to import auralab
sys.path.append(str(Path(__file__).parent.parent / "src"))


from auralab.fintech.sources.binance import BinanceFetcher
from auralab.fintech.sources.core import TradeData, TradingPair, Market
from auralab.fintech.dataset import TradingDataset


class TestBinanceFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = BinanceFetcher(cache_dir="/tmp/test_cache")

    @patch("auralab.fintech.sources.binance.urllib.request.urlretrieve")
    @patch("auralab.fintech.sources.binance.zipfile.ZipFile")
    @patch("pathlib.Path.exists")
    @patch("torch.save")
    def test_fetch_day_download(
        self, mock_save, mock_exists, mock_zipfile, mock_retrieve
    ):
        # Setup mocks
        mock_exists.return_value = False  # Force download

        # Mock ZipFile context manager and file reading
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        mock_zip.namelist.return_value = ["BTCUSDT-aggTrades-2023-01-01.csv"]

        # Mock CSV content
        # Format: agg_trade_id, price, qty, first_trade_id, last_trade_id, transact_time, is_buyer_maker
        csv_content = (
            "1,16500.0,0.001,100,100,1672531200000,True\n"
            "2,16501.0,0.002,101,101,1672531201000,False\n"
        ).encode("utf-8")

        mock_file = io.BytesIO(csv_content)
        mock_zip.open.return_value.__enter__.return_value = mock_file

        # Execute
        day = date(2023, 1, 1)
        pair = TradingPair("BTC", "USDT")
        market = Market(Market.Platform.BINANCE, "usdtm")
        data = self.fetcher.fetch_day(pair, day, market)

        # Verify download called
        mock_retrieve.assert_called_once()
        args, _ = mock_retrieve.call_args
        self.assertIn(
            "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-01-01.zip",
            args[0],
        )

        # Verify data parsing
        self.assertIsInstance(data, TradeData)
        self.assertEqual(len(data.time), 2)
        self.assertEqual(data.price[0], 16500.0)
        self.assertEqual(data.qty[1], 0.002)
        self.assertTrue(data.is_buyer_maker[0])
        self.assertFalse(data.is_buyer_maker[1])

    @patch("auralab.fintech.sources.binance.urllib.request.urlretrieve")
    @patch("auralab.fintech.sources.binance.BinanceFetcher._parse_zip")
    @patch("pathlib.Path.exists")
    def test_fetch_day_cached(self, mock_exists, mock_parse, mock_retrieve):
        mock_exists.return_value = True

        mock_data = TradeData(
            time=torch.tensor([1]),
            price=torch.tensor([1.0]),
            qty=torch.tensor([1.0]),
            is_buyer_maker=torch.tensor([True]),
        )
        mock_parse.return_value = mock_data

        pair = TradingPair("BTC", "USDT")
        market = Market(Market.Platform.BINANCE, "usdtm")
        data = self.fetcher.fetch_day(pair, date(2023, 1, 1), market)

        self.assertEqual(data, mock_data)

        # Should not call urlretrieve
        mock_retrieve.assert_not_called()

        # Should call _parse_zip
        mock_parse.assert_called_once()


class TestBinanceDataset(unittest.TestCase):
    def test_dataset_ensure(self):
        fetcher = MagicMock(spec=BinanceFetcher)
        pair = TradingPair("BTC", "USDT")
        market = Market(Market.Platform.BINANCE, "usdtm")
        dataset = TradingDataset(pair, market, fetcher)

        day = date(2023, 1, 1)
        _ = dataset.ensure(day)
        fetcher.fetch_day.assert_called_with(pair, day, market)


if __name__ == "__main__":
    unittest.main()
