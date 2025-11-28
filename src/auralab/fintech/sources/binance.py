import os
import urllib.request
import zipfile
import csv
import io
from pathlib import Path
from datetime import date, timedelta, datetime
from typing import NamedTuple, Literal, Optional, List, Dict
import logging
import torch
import numpy as np

from .core import TradeData, ExchangeFetcher, MarketType

log = logging.getLogger(__name__)

# Constants
BINANCE_SPOT_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
BINANCE_USDTM_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
BINANCE_COINM_URL = "https://data.binance.vision/data/futures/cm/daily/aggTrades"


class BinanceFetcher(ExchangeFetcher):
    """
    Fetches and caches Binance trade data.
    """

    def __init__(self, cache_dir: str | Path = ".cache/binance"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_base_url(self, market: MarketType) -> str:
        match market:
            case "spot":
                return BINANCE_SPOT_URL
            case "usdtm":
                return BINANCE_USDTM_URL
            case "coinm":
                return BINANCE_COINM_URL
            case _:
                raise ValueError(f"Unknown market: {market}")

    def fetch_day(
        self, symbol: str, day: date, market: MarketType = "usdtm"
    ) -> TradeData:
        """
        Downloads (if needed) and parses trade data for a specific day.
        """
        base_url = self._get_base_url(market)
        date_str = day.isoformat()
        filename = f"{symbol}-aggTrades-{date_str}.zip"
        url = f"{base_url}/{symbol}/{filename}"

        local_path = self.cache_dir / market / symbol / filename

        if not local_path.exists():
            log.info(f"Downloading {url} to {local_path}")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, local_path)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise FileNotFoundError(
                        f"Data not found for {symbol} on {day} at {url}"
                    ) from e
                raise

        return self._parse_zip(local_path, market)

    def _parse_zip(self, zip_path: Path, market: MarketType) -> TradeData:
        """
        Parses the zipped CSV directly into Torch tensors.
        """
        # Define column indices based on market type
        # Spot: id, price, qty, ...
        # Futures: agg_trade_id, price, qty, ...
        # Actually, the structure is usually:
        # Spot: Agg_Trade_Id, Price, Quantity, First_Trade_Id, Last_Trade_Id, Transact_Time, Is_Buyer_Maker, Best_Match
        # Futures: Agg_Trade_Id, Price, Quantity, First_Trade_Id, Last_Trade_Id, Transact_Time, Is_Buyer_Maker

        # We only care about: Transact_Time (5), Price (1), Quantity (2), Is_Buyer_Maker (6)
        # Indices are 0-based.

        idx_price = 1
        idx_qty = 2
        idx_time = 5
        idx_is_buyer_maker = 6

        times = []
        prices = []
        qtys = []
        is_buyer_makers = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_filename = zf.namelist()[0]
            with zf.open(csv_filename) as f:
                # Binance CSVs usually don't have headers in the daily dumps,
                # but sometimes they do. The code in dataframe.py suggests:
                # Spot: No header
                # Futures: Header

                wrapper = io.TextIOWrapper(f, encoding="utf-8")
                reader = csv.reader(wrapper)

                first_row = True
                for row in reader:
                    if first_row:
                        # Simple heuristic to detect header: check if first col is not a number
                        if not row[0].replace(".", "", 1).isdigit():
                            continue
                        first_row = False

                    times.append(int(row[idx_time]))
                    prices.append(float(row[idx_price]))
                    qtys.append(float(row[idx_qty]))
                    is_buyer_makers.append(
                        row[idx_is_buyer_maker] == "True"
                        or row[idx_is_buyer_maker] == "true"
                    )

        # Convert to tensors
        # Using numpy first is usually faster for list -> tensor conversion
        return TradeData(
            time=torch.from_numpy(np.array(times, dtype=np.int64)),
            price=torch.from_numpy(np.array(prices, dtype=np.float32)),
            qty=torch.from_numpy(np.array(qtys, dtype=np.float32)),
            is_buyer_maker=torch.from_numpy(np.array(is_buyer_makers, dtype=bool)),
        )
