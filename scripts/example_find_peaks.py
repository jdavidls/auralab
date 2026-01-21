#%%
"""Minimal script that showcases the Torch-based find_peaks helper."""
#%%
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from auralab.find_peaks import find_peaks


def synth_signal(num_samples: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    t = torch.linspace(0.0, 4.0 * math.pi, num_samples)
    carrier = 0.7 * torch.sin(t) + 0.3 * torch.sin(3.0 * t + 0.3)
    envelope = torch.exp(-t / (8.0 * math.pi))
    signal = envelope * carrier
    signal += 0.05 * torch.randn_like(signal)
    signal[:: num_samples // 8] += 0.4  # sprinkle accent peaks
    return signal


def plot_signal(time_axis: torch.Tensor, signal: torch.Tensor, peaks: torch.Tensor, props: dict) -> None:
    import matplotlib.pyplot as plt
    import math

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, signal, label="signal", color="tab:blue")
    plt.scatter(time_axis[peaks], signal[peaks], color="tab:red", label="peaks")

    if "left_ips" in props and "right_ips" in props:
        left_ips = props["left_ips"].detach().cpu()
        right_ips = props["right_ips"].detach().cpu()
        time_cpu = time_axis.detach().cpu()
        length = len(time_cpu) - 1 if len(time_cpu) > 1 else 1

        def idx_to_time(ip: float) -> float:
            if length <= 0:
                return float(time_cpu[0].item())
            floor_idx = max(0, min(int(math.floor(ip)), length))
            ceil_idx = min(floor_idx + 1, length)
            frac = float(ip - floor_idx)
            base = time_cpu[floor_idx].item()
            if ceil_idx == floor_idx:
                return base
            delta = time_cpu[ceil_idx].item() - base
            return base + frac * delta

        for left_ip, right_ip in zip(left_ips, right_ips):
            left_x = idx_to_time(float(left_ip))
            right_x = idx_to_time(float(right_ip))
            plt.axvspan(left_x, right_x, color="tab:red", alpha=0.1)

    plt.title("find_peaks demo")
    plt.xlabel("time (a.u.)")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Example for auralab.find_peaks")
    parser.add_argument("--samples", type=int, default=1024, help="Number of samples in the synthetic signal")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used when adding noise")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib visualization")
    args = parser.parse_args()

    signal = synth_signal(args.samples, args.seed)
    peaks, props = find_peaks(signal, height=0.2, prominence=0.25, distance=25, width=(5.0, None))

    print(f"Detected {len(peaks)} peaks")
    widths = props.get("widths")
    if widths is not None:
        widths = widths.detach().cpu()
    for idx, peak in enumerate(peaks.tolist()[:10]):
        width_val = widths[idx].item() if widths is not None and widths.numel() > idx else "n/a"
        print(f"#{idx:02d} index={peak:04d} value={signal[peak]:.3f} width={width_val}")

    if not args.no_plot:
        time_axis = torch.linspace(0.0, 1.0, args.samples)
        plot_signal(time_axis, signal, peaks, props)

if __name__ == "__main__":
    main()
