from abc import ABC, abstractmethod
from typing import NamedTuple, Literal
from datetime import date
import torch

MarketType = Literal["spot", "usdtm", "coinm"]


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
    def fetch_day(
        self, symbol: str, day: date, market: MarketType = "usdtm"
    ) -> TradeData:
        """
        Fetches trade data for a specific day.
        """
        pass
