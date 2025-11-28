#%%
"""Helpers to study constructive vs. destructive interference between a
fundamental and its multiples."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TwoToneMetrics:
    """Metrics computed for a single fundamental/ratio pair."""

    incoming_energy: float
    resultant_energy: float
    energy_gain: float
    crest_factor: float
    cross_energy_ratio: float
    envelope_contrast: float
    alignment_index: float
    rms_contrast: float


@dataclass
class InterferenceSweep:
    """Container for the sweep metrics."""

    ratios: np.ndarray
    incoming_energy: np.ndarray
    resultant_energy: np.ndarray
    energy_gain: np.ndarray
    crest_factor: np.ndarray
    cross_energy_ratio: np.ndarray
    envelope_contrast: np.ndarray
    alignment_index: np.ndarray
    rms_contrast: np.ndarray


def _time_axis(duration: float, sample_rate: float) -> np.ndarray:
    samples = int(duration * sample_rate)
    if samples <= 1:
        raise ValueError("duration * sample_rate must be > 1 sample")
    return np.linspace(0.0, duration, samples, endpoint=False)


def _signal_energy(signal: np.ndarray, dt: float) -> float:
    return float(np.sum(signal**2) * dt)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2)))


def _sine(freq: float, t: np.ndarray, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * freq * t + phase)


def _analytic_signal(signal: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(signal)
    n = signal.size
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h)


def _envelope(signal: np.ndarray) -> np.ndarray:
    return np.abs(_analytic_signal(signal))


def _windowed_rms(signal: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = max(1, window_samples)
    if window_samples > signal.size:
        return np.array([])
    window = np.ones(window_samples) / window_samples
    mean_square = np.convolve(signal**2, window, mode="valid")
    return np.sqrt(mean_square)


def compute_two_tone_metrics(
    fundamental_hz: float,
    multiple_ratio: float,
    *,
    duration: float = 1.0,
    sample_rate: float = 48_000.0,
    amplitude_f: float = 1.0,
    amplitude_g: float = 1.0,
    phase_offset: float = 0.0,
    window_duration: float = 0.02,
) -> TwoToneMetrics:
    """Return a collection of interference metrics for a two-tone mixture."""

    if fundamental_hz <= 0.0:
        raise ValueError("fundamental_hz must be positive")
    if multiple_ratio <= 0.0:
        raise ValueError("multiple_ratio must be positive")

    t = _time_axis(duration, sample_rate)
    dt = 1.0 / sample_rate

    fundamental = _sine(fundamental_hz, t, amplitude_f)
    partial = _sine(fundamental_hz * multiple_ratio, t, amplitude_g, phase_offset)
    combined = fundamental + partial

    energy_f = _signal_energy(fundamental, dt)
    energy_g = _signal_energy(partial, dt)
    energy_in = energy_f + energy_g
    energy_out = _signal_energy(combined, dt)
    gain = energy_out / energy_in if energy_in else np.nan
    crest = np.max(np.abs(combined)) / _rms(combined)

    interaction = 2.0 * np.sum(fundamental * partial) * dt
    cross_ratio = interaction / energy_in if energy_in else np.nan

    envelope = _envelope(combined)
    mean_env = float(np.mean(envelope))
    env_contrast = (
        (float(np.max(envelope)) - float(np.min(envelope))) / mean_env if mean_env else np.nan
    )

    product = fundamental * partial
    same = np.mean(product > 0)
    opposite = np.mean(product < 0)
    alignment = float(same - opposite)

    window_samples = int(window_duration * sample_rate)
    window_samples = max(1, window_samples)
    rms_vals = _windowed_rms(combined, window_samples)
    if rms_vals.size == 0:
        rms_contrast = np.nan
    else:
        min_rms = float(np.min(rms_vals))
        rms_contrast = float(np.max(rms_vals)) / min_rms if min_rms > 0 else np.inf

    return TwoToneMetrics(
        incoming_energy=float(energy_in),
        resultant_energy=float(energy_out),
        energy_gain=float(gain),
        crest_factor=float(crest),
        cross_energy_ratio=float(cross_ratio),
        envelope_contrast=float(env_contrast),
        alignment_index=float(alignment),
        rms_contrast=float(rms_contrast),
    )


def sweep_interference(
    fundamental_hz: float,
    max_ratio: float,
    *,
    num_ratios: int = 200,
    duration: float = 1.0,
    sample_rate: float = 48_000.0,
    amplitude_f: float = 1.0,
    amplitude_g: float = 1.0,
    phase_offset: float = 0.0,
    window_duration: float = 0.02,
) -> InterferenceSweep:
    """Evaluate the interference metrics for ratio linspace(1, max_ratio)."""

    ratios = np.linspace(1.0, max_ratio, num=num_ratios)
    incoming = np.empty_like(ratios)
    resulting = np.empty_like(ratios)
    gain = np.empty_like(ratios)
    crest = np.empty_like(ratios)
    cross = np.empty_like(ratios)
    envelope = np.empty_like(ratios)
    alignment = np.empty_like(ratios)
    rms = np.empty_like(ratios)

    for idx, ratio in enumerate(ratios):
        metrics = compute_two_tone_metrics(
            fundamental_hz,
            ratio,
            duration=duration,
            sample_rate=sample_rate,
            amplitude_f=amplitude_f,
            amplitude_g=amplitude_g,
            phase_offset=phase_offset,
            window_duration=window_duration,
        )
        incoming[idx] = metrics.incoming_energy
        resulting[idx] = metrics.resultant_energy
        gain[idx] = metrics.energy_gain
        crest[idx] = metrics.crest_factor
        cross[idx] = metrics.cross_energy_ratio
        envelope[idx] = metrics.envelope_contrast
        alignment[idx] = metrics.alignment_index
        rms[idx] = metrics.rms_contrast

    return InterferenceSweep(
        ratios,
        incoming,
        resulting,
        gain,
        crest,
        cross,
        envelope,
        alignment,
        rms,
    )


def plot_energy_sweep(result: InterferenceSweep) -> plt.Figure:
    """Plot incoming vs result energies plus gain."""

    fig, ax_left = plt.subplots(figsize=(8, 4))
    ax_left.plot(result.ratios, result.incoming_energy, label="E_in", color="#8888ff")
    ax_left.plot(result.ratios, result.resultant_energy, label="E_out", color="#ff8888")
    ax_left.set_xlabel("frequency multiple (G)")
    ax_left.set_ylabel("energy")
    ax_left.grid(True, alpha=0.2)
    ax_left.legend(loc="upper right")

    ax_gain = ax_left.twinx()
    ax_gain.plot(result.ratios, result.energy_gain, label="gain", color="#222222")
    ax_gain.set_ylabel("E_out / E_in")
    ax_gain.legend(loc="lower right")

    fig.tight_layout()
    return fig


def plot_crest_factor(result: InterferenceSweep, *, reference: float | None = None) -> plt.Figure:
    """Plot crest factor (peak-to-RMS) across the sweep."""

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(result.ratios, result.crest_factor, color="#268bd2", label="crest factor")
    ax.set_xlabel("frequency multiple (G)")
    ax.set_ylabel("crest = peak/RMS")
    ax.grid(True, alpha=0.3)

    if reference is not None:
        ax.axhline(reference, color="#bbbbbb", linestyle="--", label="reference")

    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_interference_metrics(result: InterferenceSweep) -> plt.Figure:
    """Plot alternative constructive/destructive metrics for comparison."""

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(result.ratios, result.cross_energy_ratio, color="#cb4b16")
    axes[0].set_ylabel("cross energy")
    axes[0].axhline(0.0, color="#aaaaaa", linewidth=0.8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.ratios, result.envelope_contrast, color="#859900")
    axes[1].set_ylabel("envelope contrast")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result.ratios, result.alignment_index, color="#6c71c4", label="alignment")
    axes[2].plot(result.ratios, result.rms_contrast, color="#d33682", label="RMS contrast")
    axes[2].set_xlabel("frequency multiple (G)")
    axes[2].set_ylabel("index / ratio")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_example_waveforms(
    fundamental_hz: float,
    *,
    multiple_ratio: float,
    duration: float = 0.02,
    sample_rate: float = 48_000.0,
    amplitude_f: float = 1.0,
    amplitude_g: float = 1.0,
    phase_offset: float = 0.0,
) -> plt.Figure:
    """Plot a short segment of each component plus the sum."""

    t = _time_axis(duration, sample_rate)
    fundamental = _sine(fundamental_hz, t, amplitude_f)
    partial = _sine(fundamental_hz * multiple_ratio, t, amplitude_g, phase_offset)
    combined = fundamental + partial

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t * 1e3, fundamental, label="fundamental")
    ax.plot(t * 1e3, partial, label=f"multiple x{multiple_ratio:.2f}")
    ax.plot(t * 1e3, combined, label="sum", linewidth=2)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    sweep = sweep_interference(
        fundamental_hz=220.0,
        max_ratio=4.0,
        num_ratios=400,
        duration=4.0,
        sample_rate=48_000.0,
        amplitude_f=1.0,
        amplitude_g=1.0,
    )
    plot_energy_sweep(sweep)
    plot_crest_factor(sweep, reference=np.sqrt(2.0))
    plot_interference_metrics(sweep)
    plot_example_waveforms(220.0, multiple_ratio=3.0)
    plt.show()
