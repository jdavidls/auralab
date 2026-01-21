"""Unit tests for the Torch-native find_peaks helper."""

import torch

from auralab.find_peaks import find_peaks


def test_basic_peak_detection():
    signal = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0])
    peaks, props = find_peaks(signal)

    expected = torch.tensor([1, 3, 5], dtype=torch.long)

    assert torch.equal(peaks, expected)
    assert torch.allclose(props["peak_heights"], signal[expected])


def test_height_threshold_and_distance_filters():
    signal = torch.tensor([0.0, 1.5, 0.0, 4.0, 0.0, 2.0, 0.0, 1.8, 0.0])

    peaks, props = find_peaks(signal, height=2.0, threshold=1.0, distance=2)

    assert torch.equal(peaks, torch.tensor([3]))
    assert torch.allclose(props["peak_heights"], torch.tensor([4.0]))
    assert torch.allclose(props["thresholds"], torch.tensor([4.0]))


def test_prominence_and_width_properties():
    signal = torch.tensor([0.0, 1.0, 3.0, 5.0, 3.0, 1.0, 0.0])

    peaks, props = find_peaks(signal, prominence=4.0, width=(2.0, None), rel_height=0.5)

    assert torch.equal(peaks, torch.tensor([3]))
    assert torch.allclose(props["prominences"], torch.tensor([5.0]))
    assert torch.allclose(props["widths"], torch.tensor([2.5]))
    assert torch.allclose(props["left_ips"], torch.tensor([1.75]))
    assert torch.allclose(props["right_ips"], torch.tensor([4.25]))
