import torch
import torch
from .sources.core import TradeData, SampledData


def sample_trades(
    trades: TradeData, step_ms: int, start_ms: int, end_ms: int
) -> SampledData:
    """
    Resamples trade data into OHLCV-like format using PyTorch operations.

    Args:
        trades: The raw trade data.
        step_ms: The sampling interval in milliseconds.
        start_ms: The start timestamp in milliseconds.
        end_ms: The end timestamp in milliseconds.

    Returns:
        SampledData containing high, low, vwap, vol, bid_vol, ask_vol.
    """

    # Calculate number of bins
    num_bins = (end_ms - start_ms) // step_ms

    device = trades.time.device

    # Filter trades within range
    # mask = (trades.time >= start_ms) & (trades.time < end_ms)

    # User requested to assert instead of filter
    if len(trades.time) > 0:
        assert (trades.time >= start_ms).all(), "Found trades before start_ms"
        assert (trades.time < end_ms).all(), "Found trades after end_ms"

    # If no trades, return empty (zeros/nans)
    if len(trades.time) == 0:
        return SampledData(
            high=torch.full(
                (num_bins,), float("nan"), device=device, dtype=torch.float32
            ),
            low=torch.full(
                (num_bins,), float("nan"), device=device, dtype=torch.float32
            ),
            vwap=torch.full(
                (num_bins,), float("nan"), device=device, dtype=torch.float32
            ),
            vol=torch.zeros(num_bins, device=device, dtype=torch.float32),
            bid_vol=torch.zeros(num_bins, device=device, dtype=torch.float32),
            ask_vol=torch.zeros(num_bins, device=device, dtype=torch.float32),
        )

    time = trades.time
    price = trades.price
    qty = trades.qty
    is_buyer_maker = trades.is_buyer_maker

    indices = (time - start_ms) // step_ms

    # 5. Volume Aggregation (Dense is fine, 0 is correct for empty)
    vol = torch.zeros(num_bins, device=device, dtype=torch.float32)
    vol.scatter_add_(0, indices, qty)

    # Bid/Ask Volume
    # Test expectation: Bid Vol = Buy Volume, Ask Vol = Sell Volume.
    # is_buyer_maker=True -> Maker was Buyer -> Taker was Seller -> Sell Order
    # is_buyer_maker=False -> Maker was Seller -> Taker was Buyer -> Buy Order

    # So we map:
    # Bid Vol -> Buy Order -> is_buyer_maker=False
    # Ask Vol -> Sell Order -> is_buyer_maker=True

    bid_mask = ~is_buyer_maker
    ask_mask = is_buyer_maker

    bid_vol = torch.zeros(num_bins, device=device, dtype=torch.float32)
    bid_vol.scatter_add_(0, indices[bid_mask], qty[bid_mask])

    ask_vol = torch.zeros(num_bins, device=device, dtype=torch.float32)
    ask_vol.scatter_add_(0, indices[ask_mask], qty[ask_mask])

    # 6. Price Aggregation (Sparse with Forward Fill)
    # Logic ported from neuralab/trading/dataframe.py

    events, inv = indices.unique(return_inverse=True)

    # Calculate 'fix' indices for forward filling
    # This maps every bin index [0..num_bins-1] to the index in 'events'
    # corresponding to the last valid event (or first if before any).

    # Note: events are sorted by unique()

    diffs = events.diff(prepend=events[0:1])
    scattered = torch.zeros(num_bins, device=device, dtype=torch.long)
    scattered.scatter_(0, events, diffs)

    cum = scattered.cumsum(dim=0)

    _, fix = cum.unique(return_inverse=True)

    # Calculate sparse metrics on 'events'
    num_events = len(events)

    # VWAP
    vwap_num = torch.zeros(num_events, device=device, dtype=torch.float32)
    vwap_num.scatter_add_(0, inv, price * qty)

    vwap_den = torch.zeros(num_events, device=device, dtype=torch.float32)
    vwap_den.scatter_add_(0, inv, qty)

    sparse_vwap = vwap_num / vwap_den

    # High / Low
    sparse_high = torch.zeros(num_events, device=device, dtype=torch.float32)
    sparse_high.scatter_reduce_(0, inv, price, reduce="amax", include_self=False)

    sparse_low = torch.zeros(num_events, device=device, dtype=torch.float32)
    sparse_low.scatter_reduce_(0, inv, price, reduce="amin", include_self=False)

    # Expand to dense using 'fix'
    vwap = sparse_vwap[fix]
    high = sparse_high[fix]
    low = sparse_low[fix]

    return SampledData(high, low, vwap, vol, bid_vol, ask_vol)
