import torch
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from auralab.sonance import (
    sonance_corr_matrix,
    measure_sonance_peaks,
)


def visualize_sonance():
    # Parameters
    f_base = 440.0
    sr = 48000
    duration = 0.2

    # Range of ratios
    # Use logarithmic spacing for better visualization of intervals?
    # Or linear to match the previous plot? Linear is fine for now, but log x-axis.
    # Define domain in Cents (logarithmic units)
    # 4 octaves: from -2400 to 2400 cents (0.25 to 4.0 ratio)
    cents = np.linspace(-2400, 2400, 4000)
    ratios = 2 ** (cents / 1200)

    # Generate Data
    print("Generating Sonance Data...")
    # Check for GPU
    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Get Correlation Matrix
    corr_matrix = sonance_corr_matrix(
        ratios, f_base=f_base, sr=sr, duration=duration, device=device
    )
    print(corr_matrix.shape)

    # Measure Sonance
    # kernel = measure_sonance_peaks(corr_matrix, correction=False)
    kernel = measure_sonance_peaks(corr_matrix)

    # Move to CPU for plotting
    corr_matrix_np = corr_matrix.cpu().numpy()
    kernel_np = kernel.cpu().numpy()

    # Limit heatmap range to relevant lags (optional, but good for visualization)
    # The valid correlation is in the first half.
    # Let's show the first 1000 samples or so, or the whole valid range?
    # duration 0.5s at 44100 is 22050 samples. That's a lot for y-axis.
    # Maybe show a zoomed in view or just the full thing?
    # Let's show the full valid range for now.
    valid_samples = corr_matrix_np.shape[1] // 2
    heatmap_data = corr_matrix_np[
        :, :valid_samples
    ].T  # Transpose so x is ratios/cents, y is lag

    # Common intervals to mark
    intervals = {
        # "??": 4 / 3,
        "Unison": 1.0,
        "m2": 16 / 15,
        "M2": 9 / 8,
        "m3": 6 / 5,
        "M3": 5 / 4,
        "P4": 4 / 3,
        "Tri": 45 / 32,
        "P5": 3 / 2,
        "m6": 8 / 5,
        "M6": 5 / 3,
        "m7": 9 / 5,
        "M7": 15 / 8,
        "Oct": 2.0,
    }

    # Prepare tick values in cents
    tick_vals = [1200 * np.log2(r) for r in intervals.values()]
    tick_text = list(intervals.keys())

    NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    note_cents = []
    note_names = []

    for step in range(-24, 25):
        note_cents.append(100 * step)
        note_names.append(f"{NOTES[step % 12]}{step // 12}")

    # Create Subplots
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,  # row_heights=[0.4, 0.6]
    )

    # # Heatmap
    # fig.add_trace(
    #     go.Heatmap(
    #         z=heatmap_data,
    #         x=cents,
    #         y=np.arange(valid_samples),
    #         colorscale="Viridis",
    #         name="Correlation Matrix",
    #     ),
    #     row=1,
    #     col=1,
    # )

    # Kernel Line Plot
    fig.add_trace(
        go.Scatter(
            x=cents,
            y=kernel_np,
            mode="lines",
            name="Sonance Kernel",
            line=dict(width=1, color="cyan"),
        ),
        row=1,
        col=1,
    )

    # Add markers for intervals (on Kernel plot)
    for name, ratio in intervals.items():
        # Calculate expected cents
        expected_cents = 1200 * np.log2(ratio)

        # Find closest index in the cents array
        idx = (np.abs(cents - expected_cents)).argmin()

        # Search for local max in a small window around the expected cents
        window = 10
        start = max(0, idx - window)
        end = min(len(cents), idx + window + 1)

        # Find max in window
        local_max_idx = start + np.argmax(kernel_np[start:end])
        val = kernel_np[local_max_idx]
        found_cents = cents[local_max_idx]

        # fig.add_vline(
        #     x=found_cents,
        #     line_width=1,
        #     line_dash="dash",
        #     line_color="gray",
        #     opacity=0.5,
        #     row=2,
        #     col=1,
        # )
        fig.add_annotation(
            x=found_cents, y=val, text=name, showarrow=True, arrowhead=1, row=1, col=1
        )

        # Also add line to heatmap for reference
        # fig.add_vline(
        #     x=found_cents,
        #     line_width=1,
        #     line_dash="dash",
        #     line_color="gray",
        #     opacity=0.3,
        #     row=1,
        #     col=1,
        # )

    fig.update_layout(
        title="Sonance Analysis: Correlation Matrix & Kernel",
        template="plotly_dark",
        height=1200,
        xaxis1=dict(
            title="Interval (Cents)",
            tickmode="array",
            tickvals=note_cents,
            ticktext=note_names,
        ),
        yaxis1=dict(title="Sonance"),
    )

    fig.show()


if __name__ == "__main__":
    visualize_sonance()
