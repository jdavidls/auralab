import torch
import logging
from datetime import date, timedelta, datetime, timezone
from typing import Literal, List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np

from .sources.core import ExchangeFetcher, MarketType, TradeData, SampledData
from .sources.binance import BinanceFetcher
from .sources.kraken import KrakenFetcher
from .sampling import sample_trades

log = logging.getLogger(__name__)


class TradingDataset:
    """
    Level 0: Manages daily raw trade data for a specific symbol and market.
    Ensures data is present locally (via fetcher).
    """

    def __init__(self, symbol: str, market: MarketType, fetcher: ExchangeFetcher):
        self.symbol = symbol
        self.market = market
        self.fetcher = fetcher

    def ensure(self, day: date) -> TradeData:
        """
        Ensures trade data for the given day is available and returns it.
        """
        return self.fetcher.fetch_day(self.symbol, day, self.market)


from concurrent.futures import ThreadPoolExecutor, as_completed


class SamplingDataset:
    """
    Level 1: Manages resampled (OHLCV) data for a single symbol/market over a time range.
    Caches resampled tensors to disk.
    """

    def __init__(
        self,
        symbol: str,
        market: MarketType,
        start_date: date,
        end_date: date,
        sample_rate_ms: int,
        cache_dir: str | Path = ".cache/sampling",
        max_workers: int = 4,
    ):
        self.symbol = symbol
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.sample_rate_ms = sample_rate_ms
        self.cache_dir = Path(cache_dir) / market / symbol
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        # Determine fetcher based on market/symbol logic or pass it in?
        # For now, let's instantiate based on market type or some registry.
        # Ideally this should be dependency injected, but for simplicity:
        if (
            "binance" in market
        ):  # e.g. binance-spot (if we had that) or just usdtm implies binance
            # The user prompt implies market is just 'usdtm', 'spot'.
            # But we need to know WHICH exchange.
            # Let's assume market strings like 'binance-usdtm' or we infer from symbol?
            # Or we pass a fetcher factory.
            # Let's use a simple mapping for now.
            if market in ["usdtm", "coinm"]:
                self.fetcher = BinanceFetcher()
            elif market == "spot":
                # Could be Binance or Kraken.
                # Let's assume Kraken for spot if symbol looks like Kraken pair?
                # Or better, let's default to Binance for spot unless specified.
                # Actually, the prompt says "tuple of symbols, tuple of markets".
                # Maybe we should pass the fetcher or exchange name.
                # Let's assume for this task: 'usdtm' -> Binance, 'spot' -> Kraken (as per our tests).
                # This is fragile. Let's try to be smarter.
                # Ideally, we should have an Exchange registry.
                if len(symbol) > 8 or "USD" in symbol:  # Heuristic
                    self.fetcher = KrakenFetcher()
                else:
                    self.fetcher = BinanceFetcher()
        else:
            # Default fallback
            self.fetcher = BinanceFetcher()

        self.trading_dataset = TradingDataset(symbol, market, self.fetcher)

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

        log.info(f"Generating sampled data for {self.symbol} {self.market}")

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

    def __init__(
        self,
        symbols: Tuple[str, ...],
        markets: Tuple[MarketType, ...],
        start_date: date,
        end_date: date,
        sample_rate: timedelta,
        batch_size: int,
        max_workers: int = 4,
    ):

        self.symbols = symbols
        self.markets = markets
        self.batch_size = batch_size
        self.sample_rate_ms = int(sample_rate.total_seconds() * 1000)

        # Load all data
        # Structure: [Symbol][Market] -> SampledData
        self.data: Dict[str, Dict[str, SampledData]] = {}

        # We need to ensure all datasets have the same length (T)
        # Since we use same start/end/rate, they should.

        self.num_steps = 0

        for symbol in symbols:
            self.data[symbol] = {}
            for market in markets:
                ds = SamplingDataset(
                    symbol,
                    market,
                    start_date,
                    end_date,
                    self.sample_rate_ms,
                    max_workers=max_workers,
                )
                sampled = ds.ensure()

                self.data[symbol][market] = sampled

                if self.num_steps == 0:
                    self.num_steps = len(sampled.high)
                elif len(sampled.high) != self.num_steps:
                    raise ValueError(
                        f"Mismatch in steps for {symbol} {market}: {len(sampled.high)} vs {self.num_steps}"
                    )

        # Stack data into tensors [T, S, M]
        # Features: High, Low, VWAP, Vol, BidVol, AskVol

        tensors = {}
        features = ["high", "low", "vwap", "vol", "bid_vol", "ask_vol"]

        for feat in features:
            # List of (S, M) tensors
            # Outer list: Symbols, Inner list: Markets
            feat_data = []
            for symbol in symbols:
                m_data = []
                for market in markets:
                    m_data.append(getattr(self.data[symbol][market], feat))
                # Stack markets: [T, M]
                feat_data.append(torch.stack(m_data, dim=1))

            # Stack symbols: [T, S, M]
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
