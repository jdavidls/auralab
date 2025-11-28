import torch
import triton
import time
import matplotlib.pyplot as plt
import os
from typing import Optional
from auralab.ema import ema


def benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmarks.")
        return

    device = torch.device("cuda")

    # Config
    batch_size = 32
    num_alphas = 1
    dim = 1

    seq_lens = [1024, 4096, 16384, 65536, 131072]
    times_triton = []
    times_torch = []
    mem_triton = []
    mem_torch = []

    print(f"Benchmarking EMA (B={batch_size}, A={num_alphas}, dim={dim})...")

    for T in seq_lens:
        print(f"  T={T}...", end="", flush=True)

        x = torch.randn(batch_size, T, device=device)
        alpha = torch.rand(num_alphas, device=device)

        # Triton
        # Warmup
        for _ in range(3):
            ema(x, alpha, dim=dim, optimized=True)
        torch.cuda.synchronize()

        # Time
        fn = lambda: ema(x, alpha, dim=dim, optimized=True)
        ret = triton.testing.do_bench(fn, warmup=10, rep=50)
        if isinstance(ret, tuple):
            ms_mean = ret[2]
        else:
            ms_mean = ret
        times_triton.append(ms_mean)

        # Memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        ema(x, alpha, dim=dim, optimized=True)
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        mem_triton.append(mem_mb)

        # Torch Reference
        # Only run if T is not too huge, otherwise it takes forever
        if T <= 16384:
            # Warmup
            for _ in range(3):
                ema(x, alpha, dim=dim, optimized=False)
            torch.cuda.synchronize()

            # Time
            start = time.time()
            for _ in range(5):
                ema(x, alpha, dim=dim, optimized=False)
            torch.cuda.synchronize()
            end = time.time()
            ms_mean_torch = (end - start) / 5 * 1000
            times_torch.append(ms_mean_torch)

            # Memory
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            ema(x, alpha, dim=dim, optimized=False)
            mem_mb_torch = torch.cuda.max_memory_allocated() / 1024 / 1024
            mem_torch.append(mem_mb_torch)
        else:
            times_torch.append(None)
            mem_torch.append(None)

        print(
            f" Triton: {ms_mean:.4f} ms / {mem_mb:.2f} MB, Torch: {times_torch[-1] if times_torch[-1] else 'N/A'} ms / {mem_torch[-1] if mem_torch[-1] else 'N/A'} MB"
        )

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Prepare valid torch data
    valid_torch_indices = [i for i, t in enumerate(times_torch) if t is not None]
    valid_seq_lens = [seq_lens[i] for i in valid_torch_indices]

    # Time Plot
    ax1.plot(seq_lens, times_triton, label="Triton EMA", marker="o")
    if valid_torch_indices:
        valid_times_torch = [times_torch[i] for i in valid_torch_indices]
        ax1.plot(valid_seq_lens, valid_times_torch, label="Torch Reference", marker="x")

    ax1.set_ylabel("Time (ms)")
    ax1.set_title("EMA Performance: Time")
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale("log")

    # Memory Plot
    ax2.plot(seq_lens, mem_triton, label="Triton EMA", marker="o")
    if valid_torch_indices:
        valid_mem_torch = [mem_torch[i] for i in valid_torch_indices]
        ax2.plot(valid_seq_lens, valid_mem_torch, label="Torch Reference", marker="x")

    ax2.set_xlabel("Sequence Length (T)")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("EMA Performance: Memory")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    plt.tight_layout()
    output_path = "doc/media/ema_benchmark.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Benchmark plot saved to {output_path}")


if __name__ == "__main__":
    benchmark()
