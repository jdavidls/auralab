import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import torch
from auralab.sonance import sonance_corr_matrix, measure_sonance_peaks, psd, acorr

# --- Data Generation ---
print("Generating Sonance Data...")
f_base = 440.0
sr = 44100
duration = 0.1
device = "cpu"  # Use CPU for simplicity in app, or check CUDA if needed

# Define domain in Cents
cents = np.linspace(-2400, 2400, 4000)
ratios = 2 ** (cents / 1200)

# Generate Correlation Matrix
# This might take a moment
corr_matrix = sonance_corr_matrix(
    ratios, f_base=f_base, sr=sr, duration=duration, device=device
)

# Measure Sonance (Weighted Sum of Peaks)
kernel = measure_sonance_peaks(corr_matrix)

# Convert to numpy for plotting
corr_matrix_np = corr_matrix.cpu().numpy()
kernel_np = kernel.cpu().numpy()
cents_np = cents
ratios_np = ratios

# Intervals for markers
intervals = {
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
# Add negative intervals? For now just positive/standard ones within range
# The plot goes from -2 octaves to +2 octaves.
# Let's add markers for +/- intervals if needed, or just standard set.

# --- Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Sonance Analysis", style={"textAlign": "center"}),
        html.Div(
            [
                dcc.Graph(id="kernel-plot"),
            ],
            style={"width": "100%", "display": "inline-block"},
        ),
        html.Div(
            [
                dcc.Graph(id="correlation-plot"),
            ],
            style={"width": "100%", "display": "inline-block"},
        ),
    ]
)


@app.callback(
    Output("kernel-plot", "figure"),
    Input("kernel-plot", "id"),  # Dummy input to trigger on load
)
def update_kernel_plot(_):
    fig = go.Figure()

    # Kernel Line
    fig.add_trace(
        go.Scatter(
            x=cents_np,
            y=kernel_np,
            mode="lines",
            name="Sonance Kernel",
            line=dict(color="cyan", width=1),
        )
    )

    # Markers
    for name, ratio in intervals.items():
        # Calculate cents for this ratio
        c = 1200 * np.log2(ratio)

        # We can plot it at +c and -c? Or just +c?
        # Let's plot at +c
        if -2400 <= c <= 2400:
            # Find closest value
            idx = (np.abs(cents_np - c)).argmin()

            # Local max search
            window = 10
            start = max(0, idx - window)
            end = min(len(cents_np), idx + window + 1)
            local_max_idx = start + np.argmax(kernel_np[start:end])
            val = kernel_np[local_max_idx]
            found_c = cents_np[local_max_idx]

            fig.add_vline(x=found_c, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_annotation(x=found_c, y=val, text=name, showarrow=True, arrowhead=1)

    # Ticks
    tick_vals = []
    tick_text = []
    NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    for step in range(-24, 25):
        tick_vals.append(100 * step)
        tick_text.append(f"{NOTES[step % 12]}{step // 12}")

    fig.update_layout(
        title="Sonance Kernel (Click to view Correlation)",
        xaxis_title="Interval (Cents)",
        yaxis_title="Sonance",
        template="plotly_dark",
        xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text),
        clickmode="event+select",
    )
    return fig


@app.callback(Output("correlation-plot", "figure"), Input("kernel-plot", "clickData"))
def update_correlation_plot(clickData):
    fig = go.Figure()

    if clickData is None:
        # Default view (e.g. Unison or 0 cents)
        selected_cents = 0
        title_suffix = "(Unison)"
    else:
        point = clickData["points"][0]
        selected_cents = point["x"]
        title_suffix = f"({selected_cents:.1f} cents)"

    # Find index
    idx = (np.abs(cents_np - selected_cents)).argmin()

    # Get correlation row
    # Valid range is first half
    valid_samples = corr_matrix_np.shape[1] // 2
    corr_row = corr_matrix_np[idx, :valid_samples]

    # print(corr_row.shape)

    # Autocorrelation
    # acorr expects 2D [batch, time]
    autocorr = (
        acorr(
            torch.as_tensor(corr_row).unsqueeze(0),
            n_fft=2 * valid_samples,
            dim=1,
            window=False,
        )
        .squeeze(0)
        .numpy()
    )
    lags = np.arange(len(corr_row))

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=corr_row,
            mode="lines",
            name="Autocorrelation",
            line=dict(color="yellow", width=1),
        )
    )

    fig.update_layout(
        title=f"Correlation Spectrum {title_suffix}",
        xaxis_title="Lag (Samples)",
        yaxis_title="Correlation",
        template="plotly_dark",
        yaxis=dict(range=[-1.1, 1.1]),
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
