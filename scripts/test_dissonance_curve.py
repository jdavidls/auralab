# %%
"""
Verification script for dissonance curve calculation.
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
from neuralab.dissonance import calculate_dissonance_curve, Partials


def plot_dissonance_curve():
    # 1. Define context: C Major chord (C4, E4, G4)
    sounding_notes_names = ["C4", "E4", "G4"]
    sounding_notes = np.array([librosa.note_to_hz(n) for n in sounding_notes_names])
    print(f"Sounding notes: {sounding_notes_names}")

    # 2. Define timbre: First 6 harmonics with 1/k amplitude
    partials = Partials.from_harmonics(n_harmonics=6, decay_exponent=1.0)
    print("Timbre: 6 harmonics, 1/k amplitude")

    # 3. Define sampling vector: From C3 to C5
    fmin = librosa.note_to_hz("C3")
    fmax = librosa.note_to_hz("C5")
    sampling_freqs = np.linspace(fmin, fmax, 1000)  # Increased resolution
    print(f"Sampling range: {fmin:.2f} Hz to {fmax:.2f} Hz")

    # 4. Calculate dissonance curve
    print("Calculating dissonance curve...")
    dissonance = calculate_dissonance_curve(sampling_freqs, sounding_notes, partials)

    # 5. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(sampling_freqs, dissonance, label="Dissonance Curve", color="cyan")

    # Mark sounding notes
    for note, freq in zip(sounding_notes_names, sounding_notes):
        plt.axvline(
            freq,
            color="magenta",
            linestyle="--",
            alpha=0.5,
            label=f"Note {note}" if note == sounding_notes_names[0] else "",
        )
        plt.text(
            freq,
            plt.ylim()[1],
            note,
            rotation=90,
            verticalalignment="bottom",
            color="white",
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Sensory Dissonance")
    plt.title("Dissonance Curve for C Major Chord (C4-E4-G4)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Use dark background as requested in guidelines
    plt.style.use("dark_background")

    output_path = "dissonance_curve_test.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_dissonance_curve()
