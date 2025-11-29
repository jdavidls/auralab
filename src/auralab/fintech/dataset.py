import os
import torch
import logging
from datetime import date, timedelta, datetime, timezone
from typing import Literal, List, Tuple, Dict, Optional, Iterable
from pathlib import Path
import numpy as np


def _parse_list(env_var: str, default: str) -> Tuple[str, ...]:
    val = os.environ.get(env_var, default)
    return tuple(s.strip() for s in val.split(",") if s.strip())


def _parse_date(env_var: str, default: str) -> date:
    val = os.environ.get(env_var, default)
    return date.fromisoformat(val)


def _parse_timedelta(env_var: str, default_minutes: int) -> timedelta:
    val = os.environ.get(env_var, str(default_minutes))
    return timedelta(minutes=int(val))


from .sources.core import (
    ExchangeFetcher,
    MarketType,
    TradeData,
    SampledData,
    TradingPair,
    Market,
)
from .sources.binance import BinanceFetcher
from .sources.kraken import KrakenFetcher
from .sampling import sample_trades

log = logging.getLogger(__name__)


class TradingDataset:
    """
    Level 0: Manages daily raw trade data for a specific symbol and market.
    Ensures data is present locally (via fetcher).
    """

    def __init__(self, pair: TradingPair, market: Market, fetcher: ExchangeFetcher):
        self.pair = pair
        self.market = market
        self.fetcher = fetcher

    def ensure(self, day: date) -> TradeData:
        """
        Ensures trade data for the given day is available and returns it.
        """
        return self.fetcher.fetch_day(self.pair, day, self.market)


from concurrent.futures import ThreadPoolExecutor, as_completed


class SamplingDataset:
    """
    Level 1: Manages resampled (OHLCV) data for a single symbol/market over a time range.
    Caches resampled tensors to disk.
    """

    def __init__(
        self,
        pair: TradingPair,
        market: Market,
        start_date: date,
        end_date: date,
        sample_rate_ms: int,
        cache_dir: str | Path = ".cache/sampling",
        max_workers: int = 4,
    ):
        self.pair = pair
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.sample_rate_ms = sample_rate_ms
        # Cache directory uses string representation of pair and market
        self.cache_dir = Path(cache_dir) / str(market) / str(pair)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        # Determine fetcher based on platform
        if market.platform == Market.Platform.BINANCE:
            self.fetcher = BinanceFetcher()
        elif market.platform == Market.Platform.KRAKEN:
            self.fetcher = KrakenFetcher()
        else:
            raise ValueError(f"Unsupported platform: {market.platform}")

        self.trading_dataset = TradingDataset(pair, market, self.fetcher)

    def _get_cache_path(self) -> Path:
        return (
            self.cache_dir
            / f"{self.start_date}_{self.end_date}_{self.sample_rate_ms}ms.pt"
        )

    def ensure(self) -> SampledData:
        cache_path = self._get_cache_path()
        if cache_path.exists():
            log.info(f"Loading sampled data from {cache_path}")
            # Allow SampledData to be loaded
            torch.serialization.add_safe_globals([SampledData])
            return torch.load(cache_path, weights_only=True)

        log.info(f"Generating sampled data for {self.pair} {self.market}")

        # Load all days in parallel
        days = []
        curr = self.start_date
        while curr < self.end_date:
            days.append(curr)
            curr += timedelta(days=1)

        all_trades_map = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_day = {
                executor.submit(self.trading_dataset.ensure, day): day for day in days
            }

            for future in as_completed(future_to_day):
                day = future_to_day[future]
                try:
                    trades = future.result()
                    all_trades_map[day] = trades
                except Exception as e:
                    log.error(f"Failed to fetch data for {day}: {e}")
                    raise e

        # Sort trades by day to ensure correct order
        all_trades = [all_trades_map[day] for day in sorted(all_trades_map.keys())]

        # Merge trades
        # We need to concatenate tensors
        # TradeData is a NamedTuple of tensors

        time = torch.cat([t.time for t in all_trades])
        price = torch.cat([t.price for t in all_trades])
        qty = torch.cat([t.qty for t in all_trades])
        is_buyer_maker = torch.cat([t.is_buyer_maker for t in all_trades])

        merged_trades = TradeData(time, price, qty, is_buyer_maker)

        # Sample
        start_ms = int(
            datetime.combine(
                self.start_date, datetime.min.time(), tzinfo=timezone.utc
            ).timestamp()
            * 1000
        )
        end_ms = int(
            datetime.combine(
                self.end_date, datetime.min.time(), tzinfo=timezone.utc
            ).timestamp()
            * 1000
        )

        sampled = sample_trades(merged_trades, self.sample_rate_ms, start_ms, end_ms)

        # Cache
        torch.save(sampled, cache_path)
        return sampled


class FintechDataset(torch.utils.data.Dataset):
    """
    Level 2: Top-level dataset providing aligned tensors [T, S, M, F] for training.
    """

    DEFAULT_SYMBOLS = _parse_list("FINTECH_SYMBOLS", "BTC-USDT,ETH-USDT,ETH-BTC")
    DEFAULT_MARKETS = _parse_list("FINTECH_MARKETS", "binance-usdtm")
    DEFAULT_START_DATE = _parse_date("FINTECH_START_DATE", "2023-01-01")
    DEFAULT_END_DATE = _parse_date("FINTECH_END_DATE", "2023-02-01")
    DEFAULT_SAMPLE_RATE = _parse_timedelta("FINTECH_SAMPLE_RATE_MIN", 1)
    DEFAULT_BATCH_SIZE = int(os.environ.get("FINTECH_BATCH_SIZE", 64))

    @classmethod
    def default(cls):
        return cls()

    def __init__(
        self,
        symbols: str | Iterable[str | TradingPair] = DEFAULT_SYMBOLS,
        markets: str | Iterable[str | Market] = DEFAULT_MARKETS,
        start_date: date = DEFAULT_START_DATE,
        end_date: date = DEFAULT_END_DATE,
        sample_rate: timedelta = DEFAULT_SAMPLE_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_workers: int = 4,
    ):

        self.batch_size = batch_size
        self.sample_rate_ms = int(sample_rate.total_seconds() * 1000)

        # Parse markets
        if isinstance(markets, str):
            self.markets = Market.many(markets)
        else:
            # Filter out already parsed objects to avoid passing them to parse_many which expects strings
            # This restores support for passing Market objects directly
            strs = [m for m in markets if isinstance(m, str)]
            objs = [m for m in markets if isinstance(m, Market)]
            self.markets = objs + Market.many(*strs)

        # Parse symbols
        if isinstance(symbols, str):
            self.pairs = TradingPair.many(symbols)
        else:
            strs = [s for s in symbols if isinstance(s, str)]
            objs = [s for s in symbols if isinstance(s, TradingPair)]
            self.pairs = objs + TradingPair.many(*strs)

        # Extract unique assets and create mapping
        self.assets = sorted(
            list(set([p.base for p in self.pairs] + [p.quote for p in self.pairs]))
        )
        self.asset_to_idx = {a: i for i, a in enumerate(self.assets)}

        # Store indices for each pair [S, 2] (Base, Quote)
        self.pair_indices = torch.tensor(
            [
                [self.asset_to_idx[p.base], self.asset_to_idx[p.quote]]
                for p in self.pairs
            ],
            dtype=torch.long,
        )

        # Load all data
        # Structure: [PairIndex][MarketIndex] -> SampledData
        self.data: Dict[int, Dict[int, SampledData]] = {}

        self.num_steps = 0

        for i, pair in enumerate(self.pairs):
            self.data[i] = {}
            for j, market in enumerate(self.markets):
                ds = SamplingDataset(
                    pair,
                    market,
                    start_date,
                    end_date,
                    self.sample_rate_ms,
                    max_workers=max_workers,
                )
                sampled = ds.ensure()

                self.data[i][j] = sampled

                if self.num_steps == 0:
                    self.num_steps = len(sampled.high)
                elif len(sampled.high) != self.num_steps:
                    raise ValueError(
                        f"Mismatch in steps for {pair} {market}: {len(sampled.high)} vs {self.num_steps}"
                    )

        # Stack data into tensors [T, S, M]
        # Features: High, Low, VWAP, Vol, BidVol, AskVol

        tensors = {}
        features = ["high", "low", "vwap", "vol", "bid_vol", "ask_vol"]

        for feat in features:
            # List of (S, M) tensors
            # Outer list: Pairs (S), Inner list: Markets (M)
            feat_data = []
            for i in range(len(self.pairs)):
                m_data = []
                for j in range(len(self.markets)):
                    m_data.append(getattr(self.data[i][j], feat))
                # Stack markets: [T, M]
                feat_data.append(torch.stack(m_data, dim=1))

            # Stack pairs: [T, S, M]
            tensors[feat] = torch.stack(feat_data, dim=1)

        self.dataset = SampledData(**tensors)

    def __len__(self):
        return self.num_steps // self.batch_size

    def __getitem__(self, idx) -> SampledData:
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        # Return slice of all tensors wrapped in SampledData
        return SampledData(
            high=self.dataset.high[start:end],
            low=self.dataset.low[start:end],
            vwap=self.dataset.vwap[start:end],
            vol=self.dataset.vol[start:end],
            bid_vol=self.dataset.bid_vol[start:end],
            ask_vol=self.dataset.ask_vol[start:end],
        )
