# %%
"""
Script to verify dissonance curve calculation using Plotly.
Samples frequencies corresponding to piano keys.
"""
import sys
from pathlib import Path

# Ensure src is in path if running as script
project_root = Path(__file__).resolve().parents[1]
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))

import numpy as np
import plotly.graph_objects as go
import librosa
from neuralab.dissonance import calculate_dissonance_curve, Partials


def get_piano_frequencies():
    """Returns frequencies for standard 88-key piano (A0 to C8)."""
    # MIDI notes 21 (A0) to 108 (C8)
    midi_notes = np.arange(21, 109)
    return librosa.midi_to_hz(midi_notes)


def plot_dissonance_check():
    # 1. Define Context: C Major Chord (C4, E4, G4)
    sounding_notes_names = ["C4", "E4", "G4"]
    sounding_notes = np.array([librosa.note_to_hz(n) for n in sounding_notes_names])

    # 2. Define Timbre
    # Using a simple harmonic timbre
    partials = Partials.from_harmonics(n_harmonics=6, decay_exponent=1.0)

    # 3. Define Sampling Frequencies (Piano Keys)
    # We want a continuous curve for visualization, but we'll mark the piano keys
    # Let's sample continuously from A0 to C8 for the curve

    fmin = librosa.note_to_hz("A0")
    fmax = librosa.note_to_hz("C8")
    sampling_freqs_continuous = np.logspace(np.log10(fmin), np.log10(fmax), 2000)

    # Also get exact piano key frequencies for markers
    piano_freqs = get_piano_frequencies()
    piano_note_names = [librosa.midi_to_note(m) for m in np.arange(21, 109)]

    # 4. Calculate Dissonance
    print("Calculating dissonance curve...")
    dissonance_continuous = calculate_dissonance_curve(
        sampling_freqs_continuous, sounding_notes, partials
    )
    dissonance_piano = calculate_dissonance_curve(piano_freqs, sounding_notes, partials)

    # 5. Plot with Plotly
    fig = go.Figure()

    # Continuous Curve
    fig.add_trace(
        go.Scatter(
            x=sampling_freqs_continuous,
            y=dissonance_continuous,
            mode="lines",
            name="Dissonance Curve",
            line=dict(color="#00FFFF", width=2),  # Cyan
        )
    )

    # Piano Keys Markers
    fig.add_trace(
        go.Scatter(
            x=piano_freqs,
            y=dissonance_piano,
            mode="markers",
            name="Piano Keys",
            marker=dict(size=6, color="#FF00FF", symbol="circle"),  # Magenta
            text=piano_note_names,
            hovertemplate="<b>%{text}</b><br>Freq: %{x:.2f} Hz<br>Diss: %{y:.4f}<extra></extra>",
        )
    )

    # Mark Sounding Notes
    for note, freq in zip(sounding_notes_names, sounding_notes):
        fig.add_vline(
            x=freq, line_width=1, line_dash="dash", line_color="white", opacity=0.5
        )
        fig.add_annotation(
            x=freq, y=max(dissonance_continuous), text=note, showarrow=False, yshift=10
        )

    # Layout
    fig.update_layout(
        title="Sensory Dissonance Curve (C Major Context)",
        xaxis=dict(
            type="log", title="Frequency (Hz)", range=[np.log10(fmin), np.log10(fmax)]
        ),
        yaxis_title="Dissonance",
        template="plotly_dark",
        hovermode="closest",
    )

    # Save to HTML
    output_file = project_root / "notebooks" / "dissonance_check.html"
    # fig.write_html(str(output_file))
    # print(f"Plot saved to {output_file}")

    # Also try to show if interactive (optional, depends on env)
    fig.show()


if __name__ == "__main__":
    plot_dissonance_check()
