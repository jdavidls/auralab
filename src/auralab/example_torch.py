# %%
"""Example of using the PyTorch CQT implementation."""

from pathlib import Path
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from neuralab.torch_cqt import cqt


def plot_torch_cqt(
    cqt_tensor: torch.Tensor, sr: int, hop_length: int, bpo: int, title: str = "CQT"
):
    """
    Plot the magnitude of a complex CQT tensor.
    """
    # Convert to magnitude
    # cqt_tensor shape is (n_bins, time, 2) for complex
    if cqt_tensor.shape[-1] == 2:
        mag = torch.sqrt(cqt_tensor[..., 0] ** 2 + cqt_tensor[..., 1] ** 2)
    else:
        mag = cqt_tensor

    # Convert to numpy for plotting
    mag_np = mag.detach().cpu().numpy()

    # Convert to dB
    C_db = librosa.amplitude_to_db(mag_np, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

    img = librosa.display.specshow(
        C_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="cqt_note",
        bins_per_octave=bpo,
        ax=ax,
    )

    output_file = project_root / "data" / "torch_cqt.png"
    fig.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    audio_path = project_root / "data" / "guitar.mp3"

    if not audio_path.exists():
        print(f"Audio file not found at {audio_path}")
        # Create a dummy signal if file doesn't exist
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4
        print("Created dummy sine wave signal")
    else:
        print(f"Loading {audio_path}")
        y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # Parameters
    hop_length = 128
    bins_per_octave = 48
    n_bins = bins_per_octave * 9
    fmin = librosa.note_to_hz("C0")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Compute CQT using our new torch implementation
    print("Computing CQT with PyTorch...")
    C_torch = cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        device=device,
    )

    print(f"CQT shape: {C_torch.shape}")

    # Plot
    plot_torch_cqt(
        C_torch, sr, hop_length, bins_per_octave, title="PyTorch CQT Magnitude"
    )
