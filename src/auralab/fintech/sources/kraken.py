import urllib.request
import urllib.error
import json
import time
import logging
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from typing import List, Tuple
import torch
import numpy as np

from .core import TradeData, ExchangeFetcher, MarketType

log = logging.getLogger(__name__)

KRAKEN_PUBLIC_URL = "https://api.kraken.com/0/public/Trades"


class KrakenFetcher(ExchangeFetcher):
    """
    Fetches and caches Kraken trade data using the public REST API.
    """

    def __init__(
        self, cache_dir: str | Path = ".cache/kraken", rate_limit_delay: float = 1.0
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay

    def _get_pair_name(self, symbol: str) -> str:
        # Kraken uses specific pair names, e.g., XBTUSD for BTC/USD
        # For simplicity, we might expect the user to pass the Kraken pair name
        # or map common ones. Let's assume the user passes the Kraken pair name.
        return symbol

    def fetch_day(
        self, symbol: str, day: date, market: MarketType = "spot"
    ) -> TradeData:
        """
        Fetches trade data for a specific day.
        Note: Kraken API is pagination-based. Fetching a whole day might take multiple requests.
        """
        if market != "spot":
            raise NotImplementedError(
                "Only spot market is currently supported for Kraken"
            )

        pair = self._get_pair_name(symbol)
        cache_file = self.cache_dir / f"{pair}-{day.isoformat()}.pt"

        if cache_file.exists():
            log.info(f"Loading cached data from {cache_file}")
            return torch.load(cache_file)

        log.info(f"Fetching Kraken data for {pair} on {day}")

        # Kraken 'since' is in nanoseconds? No, documentation says "id" which is time based.
        # Actually, for 'Trades', 'since' is a timestamp in nanoseconds (or seconds depending on context, usually ns for high precision).
        # Let's verify: Kraken API docs say "since" is "id" of the last trade.
        # But passing a timestamp usually works to start.

        start_dt = datetime.combine(day, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        end_dt = start_dt + timedelta(days=1)

        # Convert to nanoseconds for Kraken API
        since = int(start_dt.timestamp() * 1_000_000_000)
        end_ts = int(end_dt.timestamp() * 1_000_000_000)

        all_trades = []

        while True:
            url = f"{KRAKEN_PUBLIC_URL}?pair={pair}&since={since}"
            log.debug(f"Requesting {url}")

            try:
                with urllib.request.urlopen(url) as response:
                    data = json.loads(response.read().decode())
            except urllib.error.HTTPError as e:
                log.error(f"HTTP Error: {e.code} {e.reason}")
                raise

            if data.get("error"):
                raise RuntimeError(f"Kraken API Error: {data['error']}")

            result = data["result"]
            # The key for trades is the pair name, but it might differ slightly from request (e.g. XXBTZUSD)
            # We find the key that is not 'last'
            pair_key = next(k for k in result.keys() if k != "last")
            trades = result[pair_key]

            if not trades:
                break

            # Filter trades within the day and append
            # Trade format: [price, volume, time, buy/sell, market/limit, miscellaneous, trade_id]
            # Time is in seconds (float)

            last_trade_time_ns = 0

            for trade in trades:
                # trade[2] is time in seconds
                trade_ts_ns = int(float(trade[2]) * 1_000_000_000)

                if trade_ts_ns >= end_ts:
                    # We went past the end of the day
                    # But we must be careful, 'since' returns trades *after* that id.
                    # If we are strictly > end_ts, we can stop.
                    pass

                if trade_ts_ns < end_ts:
                    all_trades.append(trade)

                last_trade_time_ns = trade_ts_ns

            # Update 'since' for next page
            new_since = int(result["last"])

            if new_since == since:
                break  # No new data

            since = new_since

            # Check if we are done with this day
            # If the last batch contained trades beyond end_ts, we are definitely done.
            # Or if the 'last' pointer is beyond end_ts.
            if since >= end_ts:
                break

            time.sleep(self.rate_limit_delay)

        # Parse accumulated trades
        # [price, volume, time, buy/sell, market/limit, miscellaneous, trade_id]
        # price: string
        # volume: string
        # time: float (seconds)
        # buy/sell: 'b' or 's'
        # market/limit: 'm' or 'l'

        times = []
        prices = []
        qtys = []
        is_buyer_makers = (
            []
        )  # Kraken doesn't explicitly say "maker", but 'b'/'s' indicates who initiated?
        # Actually 'b' = buy, 's' = sell (side of the taker usually).
        # If side is 'b', the taker bought, so the maker was a seller.
        # If side is 's', the taker sold, so the maker was a buyer.
        # Wait, 'is_buyer_maker' in Binance means "Was the maker a buyer?"
        # If trade is 'sell' (taker sold), then maker bought -> is_buyer_maker = True.
        # If trade is 'buy' (taker bought), then maker sold -> is_buyer_maker = False.

        for t in all_trades:
            times.append(int(float(t[2]) * 1000))  # ms
            prices.append(float(t[0]))
            qtys.append(float(t[1]))

            side = t[3]
            # is_buyer_maker logic:
            # side 'b' -> taker buy -> maker sell -> is_buyer_maker = False
            # side 's' -> taker sell -> maker buy -> is_buyer_maker = True
            is_buyer_makers.append(side == "s")

        trade_data = TradeData(
            time=torch.tensor(times, dtype=torch.int64),
            price=torch.tensor(prices, dtype=torch.float32),
            qty=torch.tensor(qtys, dtype=torch.float32),
            is_buyer_maker=torch.tensor(is_buyer_makers, dtype=torch.bool),
        )

        # Cache the result
        log.info(f"Saving {len(times)} trades to {cache_file}")
        torch.save(trade_data, cache_file)

        return trade_data
