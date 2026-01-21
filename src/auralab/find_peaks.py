"""Torch-native helpers inspired by :func:`scipy.signal.find_peaks`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Optional, Sequence, Tuple

import torch


TensorDict = Dict[str, torch.Tensor]


def find_peaks(
	signal: torch.Tensor | Sequence[float],
	*,
	height: Optional[Tuple[Optional[float], Optional[float]] | float] = None,
	threshold: Optional[Tuple[Optional[float], Optional[float]] | float] = None,
	distance: Optional[int] = None,
	prominence: Optional[Tuple[Optional[float], Optional[float]] | float] = None,
	width: Optional[Tuple[Optional[float], Optional[float]] | float] = None,
	rel_height: float = 0.5,
	wlen: Optional[int] = None,
) -> Tuple[torch.Tensor, TensorDict]:
	"""Detect 1-D peaks using a SciPy-like interface backed by PyTorch ops.

	Parameters roughly follow :func:`scipy.signal.find_peaks`, but the current
	implementation focuses on the options most commonly used in this project
	(``height``, ``threshold``, ``distance``, ``prominence`` and ``width``).
	The function is device- and dtype-aware, so tensors on CUDA will stay on
	the GPU for the bulk of the computation.
	"""

	x = _ensure_1d_tensor(signal)
	if x.numel() < 3:
		return torch.empty(0, dtype=torch.long, device=x.device), {}

	peaks = _candidate_peaks(x)
	if peaks.numel() == 0:
		return peaks, {}

	peak_heights = x[peaks]
	properties: TensorDict = {"peak_heights": peak_heights}

	def _subset(indices: torch.Tensor) -> None:
		nonlocal peaks, peak_heights, properties
		peaks = peaks[indices]
		peak_heights = peak_heights[indices]
		for key, value in properties.items():
			properties[key] = value[indices.to(value.device)]
		properties["peak_heights"] = peak_heights

	def _apply_mask(mask: torch.Tensor) -> None:
		nonlocal peaks, peak_heights, properties
		if mask.numel() == 0 or bool(mask.all().item()):
			return
		if not bool(mask.any().item()):
			peaks = torch.empty(0, dtype=torch.long, device=x.device)
			peak_heights = torch.empty(0, dtype=x.dtype, device=x.device)
			for key in list(properties.keys()):
				prop = properties[key]
				properties[key] = torch.empty(0, dtype=prop.dtype, device=prop.device)
			properties["peak_heights"] = peak_heights
			return
		indices = torch.nonzero(mask, as_tuple=False).view(-1)
		_subset(indices)

	# Height filtering -----------------------------------------------------
	h_min, h_max = _parse_bounds(height, x)
	if h_min is not None or h_max is not None:
		mask = torch.ones_like(peak_heights, dtype=torch.bool)
		if h_min is not None:
			mask &= peak_heights >= h_min
		if h_max is not None:
			mask &= peak_heights <= h_max
		_apply_mask(mask)
		if peaks.numel() == 0:
			return peaks, properties

	# Threshold filtering --------------------------------------------------
	t_min, t_max = _parse_bounds(threshold, x)
	if (t_min is not None or t_max is not None) and peaks.numel() > 0:
		left_diff = peak_heights - x[peaks - 1]
		right_diff = peak_heights - x[peaks + 1]
		edge_diff = torch.minimum(left_diff, right_diff)
		properties["thresholds"] = edge_diff
		mask = torch.ones_like(edge_diff, dtype=torch.bool)
		if t_min is not None:
			mask &= edge_diff >= t_min
		if t_max is not None:
			mask &= edge_diff <= t_max
		_apply_mask(mask)
		if peaks.numel() == 0:
			return peaks, properties

	# Distance filtering ---------------------------------------------------
	if distance is not None and peaks.numel() > 1:
		distance = int(distance)
		if distance > 0:
			order = torch.argsort(peak_heights, descending=True)
			keep: list[int] = []
			kept_positions: list[int] = []
			for idx in order.detach().cpu().tolist():
				candidate = int(peaks[idx].item())
				if all(abs(candidate - pos) > distance for pos in kept_positions):
					keep.append(idx)
					kept_positions.append(candidate)
			if keep:
				keep_tensor = torch.tensor(keep, dtype=torch.long, device=peaks.device)
				keep_tensor, _ = torch.sort(keep_tensor)
				_subset(keep_tensor)
			else:
				peaks = peaks.new_empty(0)
				peak_heights = peak_heights.new_empty(0)
				for key in list(properties.keys()):
					properties[key] = properties[key].new_empty(0)
				properties["peak_heights"] = peak_heights
				return peaks, properties

	# Prominence -----------------------------------------------------------
	need_prominence = prominence is not None or width is not None
	if peaks.numel() > 0 and need_prominence:
		prominences, left_bases, right_bases = _compute_prominence(x, peaks, wlen)
		properties.update(
			{
				"prominences": prominences,
				"left_bases": left_bases,
				"right_bases": right_bases,
			}
		)

		p_min, p_max = _parse_bounds(prominence, x)
		if p_min is not None or p_max is not None:
			mask = torch.ones_like(prominences, dtype=torch.bool)
			if p_min is not None:
				mask &= prominences >= p_min
			if p_max is not None:
				mask &= prominences <= p_max
			_apply_mask(mask)
			if peaks.numel() == 0:
				return peaks, properties

	# Width ----------------------------------------------------------------
	if peaks.numel() > 0 and width is not None:
		if "prominences" not in properties:
			prominences, left_bases, right_bases = _compute_prominence(x, peaks, wlen)
			properties.update(
				{
					"prominences": prominences,
					"left_bases": left_bases,
					"right_bases": right_bases,
				}
			)
		widths, left_ips, right_ips = _compute_widths(x, peaks, properties["prominences"], rel_height)
		properties.update(
			{
				"widths": widths,
				"left_ips": left_ips,
				"right_ips": right_ips,
			}
		)

		w_min, w_max = _parse_bounds(width, x)
		mask = torch.ones_like(widths, dtype=torch.bool)
		if w_min is not None:
			mask &= widths >= w_min
		if w_max is not None:
			mask &= widths <= w_max
		_apply_mask(mask)

	return peaks, properties


def _ensure_1d_tensor(signal: torch.Tensor | Sequence[float]) -> torch.Tensor:
	x = signal if torch.is_tensor(signal) else torch.as_tensor(signal)
	if x.ndim != 1:
		raise ValueError("find_peaks expects a 1-D input tensor")
	if not torch.is_floating_point(x):
		x = x.to(torch.float32)
	return x


def _candidate_peaks(x: torch.Tensor) -> torch.Tensor:
	if x.numel() < 3:
		return torch.empty(0, dtype=torch.long, device=x.device)
	left = x[:-2]
	mid = x[1:-1]
	right = x[2:]
	mask = (mid > left) & (mid > right)
	return torch.nonzero(mask, as_tuple=False).view(-1) + 1


def _parse_bounds(bounds: Optional[Tuple[Optional[float], Optional[float]] | float], x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
	if bounds is None:
		return None, None
	device = x.device
	dtype = x.dtype
	if isinstance(bounds, torch.Tensor):
		if bounds.ndim == 0 or bounds.numel() == 1:
			low = bounds.item()
			high = None
		else:
			flat = bounds.flatten().tolist()
			low = flat[0]
			high = flat[1] if len(flat) > 1 else None
	elif isinstance(bounds, Iterable) and not isinstance(bounds, (float, int)):
		seq = list(bounds)
		low = seq[0] if len(seq) > 0 else None
		high = seq[1] if len(seq) > 1 else None
	else:
		low = bounds
		high = None

	def _maybe_tensor(value):
		if value is None:
			return None
		return torch.as_tensor(value, dtype=dtype, device=device)

	return _maybe_tensor(low), _maybe_tensor(high)


def _compute_prominence(
	x: torch.Tensor,
	peaks: torch.Tensor,
	wlen: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	length = x.numel()
	if wlen is not None:
		wlen = max(int(wlen), 1)
	half_window = None if wlen is None else max(wlen // 2, 1)

	prominences = torch.empty_like(peaks, dtype=x.dtype)
	left_bases = torch.empty_like(peaks)
	right_bases = torch.empty_like(peaks)

	peak_indices = peaks.detach().cpu().tolist()
	for idx, peak in enumerate(peak_indices):
		left_limit = 0 if half_window is None else max(peak - half_window, 0)
		right_limit = length - 1 if half_window is None else min(peak + half_window, length - 1)

		left_slice = x[left_limit : peak + 1]
		right_slice = x[peak : right_limit + 1]
		left_min_val, left_rel_idx = torch.min(left_slice, dim=0)
		right_min_val, right_rel_idx = torch.min(right_slice, dim=0)

		left_idx = left_limit + int(left_rel_idx.item())
		right_idx = peak + int(right_rel_idx.item())
		base_val = torch.maximum(left_min_val, right_min_val)

		prominences[idx] = x[peak] - base_val
		left_bases[idx] = left_idx
		right_bases[idx] = right_idx

	return prominences, left_bases, right_bases


def _compute_widths(
	x: torch.Tensor,
	peaks: torch.Tensor,
	prominences: torch.Tensor,
	rel_height: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	rel_height = float(rel_height)
	widths = torch.empty_like(peaks, dtype=x.dtype)
	left_ips = torch.empty_like(peaks, dtype=x.dtype)
	right_ips = torch.empty_like(peaks, dtype=x.dtype)

	peak_indices = peaks.detach().cpu().tolist()
	for idx, peak in enumerate(peak_indices):
		level = x[peak] - prominences[idx] * rel_height
		left = peak
		while left > 0 and x[left] > level:
			left -= 1
		right = peak
		while right < x.numel() - 1 and x[right] > level:
			right += 1

		left_ip = _interpolate_ip(x[left], x[left + 1], left, level) if left < peak else float(left)
		right_ip = _interpolate_ip(x[right - 1], x[right], right - 1, level) if right > peak else float(right)

		widths[idx] = right_ip - left_ip
		left_ips[idx] = left_ip
		right_ips[idx] = right_ip

	return widths, left_ips, right_ips


def _interpolate_ip(v0: torch.Tensor, v1: torch.Tensor, idx: int, level: torch.Tensor) -> float:
	start = _to_python_float(v0)
	end = _to_python_float(v1)
	lvl = _to_python_float(level)
	denom = end - start
	if denom == 0:
		return float(idx)
	return float(idx) + (lvl - start) / denom


def _to_python_float(value: torch.Tensor | float | int) -> float:
	if isinstance(value, torch.Tensor):
		return float(value.detach().cpu().item())
	return float(value)


