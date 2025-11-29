from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal, NamedTuple, List, Iterable, Union, Self
from datetime import date
import torch

MarketType = Literal["spot", "usdtm", "coinm"]
Asset = str


class Market(NamedTuple):
    class Platform(str, Enum):
        BINANCE = "binance"
        KRAKEN = "kraken"

    platform: Platform
    type: str  # "spot", "usdtm", "coinm"

    def __str__(self) -> str:
        return f"{self.platform}-{self.type}"

    @classmethod
    def from_str(cls, s: str) -> "Market":
        if "-" not in s:
            raise ValueError(
                f"Invalid market string: {s}. Use 'platform-type' format (e.g. 'binance-spot')."
            )
        platform, type_ = s.split("-", 1)
        return cls(cls.Platform(platform), type_)

    @classmethod
    def many(cls, *input: str) -> list["Market"]:
        return [cls.from_str(s) for i in input for s in i.split(",")]


class TradingPair(NamedTuple):
    base: Asset
    quote: Asset

    def __str__(self) -> str:
        return f"{self.base}-{self.quote}"

    @classmethod
    def from_str(cls, s: str) -> "TradingPair":
        if "-" not in s:
            raise ValueError(
                f"Cannot parse symbol string: {s}. Use TradingPair or 'BASE-QUOTE' format."
            )
        base, quote = s.split("-")
        return cls(base, quote)

    @classmethod
    def many(cls, *input: str) -> List["TradingPair"]:
        return [cls.from_str(s) for i in input for s in i.split(",")]


class TradeData(NamedTuple):
    """
    Container for trade data as PyTorch tensors.
    """

    time: torch.Tensor  # int64 (milliseconds)
    price: torch.Tensor  # float32
    qty: torch.Tensor  # float32
    is_buyer_maker: torch.Tensor  # bool


class SampledData(NamedTuple):
    """
    Container for resampled trade data (OHLCV).
    """

    high: torch.Tensor  # float32
    low: torch.Tensor  # float32
    vwap: torch.Tensor  # float32
    vol: torch.Tensor  # float32
    bid_vol: torch.Tensor  # float32
    ask_vol: torch.Tensor  # float32


class ExchangeFetcher(ABC):
    """
    Abstract base class for fetching trade data from different exchanges.
    """

    @abstractmethod
    def format_symbol(self, pair: TradingPair) -> str:
        """
        Formats a TradingPair into the exchange-specific symbol string.
        """
        pass

    @abstractmethod
    def fetch_day(self, pair: TradingPair, day: date, market: Market) -> TradeData:
        """
        Fetches trade data for a specific day.
        """
        pass
