# %%
"""
Interactive Dash App for Dissonance Exploration.
"""
import sys
from pathlib import Path

# Ensure src is in path
project_root = Path(__file__).resolve().parents[1]
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))

import numpy as np
import librosa
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from auralab.dissonance import calculate_dissonance_curve, Partials


# --- Data Setup ---
def get_piano_frequencies():
    midi_notes = np.arange(21, 109)
    return librosa.midi_to_hz(midi_notes)


# Generate note options for full piano range (A0 to C8)
# MIDI: A0=21, C8=108
note_options = []
for m in range(21, 109):
    note_name = librosa.midi_to_note(m)
    note_options.append({"label": note_name, "value": note_name})

# Pre-calculate sampling range
fmin = librosa.note_to_hz("A0")
fmax = librosa.note_to_hz("C8")
sampling_freqs_continuous = np.logspace(np.log10(fmin), np.log10(fmax), 2000)
piano_freqs = get_piano_frequencies()
piano_note_names = [librosa.midi_to_note(m) for m in np.arange(21, 109)]

# Timbre (Fixed for now)
partials = Partials.from_harmonics(n_harmonics=10, decay_exponent=1.0)
# partials = Partials.flat(n_harmonics=6)

# --- App Setup ---
app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "Interactive Sensory Dissonance Explorer",
            style={"textAlign": "center", "color": "#ffffff"},
        ),
        html.Div(
            [
                html.Label(
                    "Select Sounding Notes (Context):",
                    style={"color": "#ffffff", "fontSize": "18px"},
                ),
                dcc.Dropdown(
                    id="note-selector",
                    options=note_options,
                    value=["C4"],  # Default C Major
                    multi=True,
                    style={"color": "#000000"},  # Text color inside dropdown
                ),
            ],
            style={"width": "50%", "margin": "auto", "padding": "20px"},
        ),
        dcc.Graph(id="dissonance-graph", style={"height": "70vh"}),
        html.Div(id="status-output", style={"textAlign": "center", "color": "#aaaaaa"}),
    ],
    style={"backgroundColor": "#111111", "minHeight": "100vh", "padding": "20px"},
)


@app.callback(Output("dissonance-graph", "figure"), Input("note-selector", "value"))
def update_graph(selected_notes):
    if not selected_notes:
        # Return empty plot or default
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", title="Select notes to see dissonance curve"
        )
        return fig

    # Convert notes to Hz
    sounding_notes = np.array([librosa.note_to_hz(n) for n in selected_notes])

    # Calculate Dissonance
    dissonance_continuous = calculate_dissonance_curve(
        sampling_freqs_continuous, sounding_notes, partials
    )
    dissonance_piano = calculate_dissonance_curve(piano_freqs, sounding_notes, partials)

    # Calculate Dissonance at Sounding Notes
    dissonance_at_notes = calculate_dissonance_curve(
        sounding_notes, sounding_notes, partials
    )
    total_sounding_dissonance = np.sum(dissonance_at_notes)

    # Plot
    fig = go.Figure()

    # Continuous Curve
    fig.add_trace(
        go.Scatter(
            x=sampling_freqs_continuous,
            y=dissonance_continuous,
            mode="lines",
            name="Dissonance Curve",
            line=dict(color="#00FFFF", width=2),
        )
    )

    # Piano Keys Markers
    fig.add_trace(
        go.Scatter(
            x=piano_freqs,
            y=dissonance_piano,
            mode="markers",
            name="Piano Keys",
            marker=dict(size=6, color="#FF00FF", symbol="circle"),
            text=piano_note_names,
            hovertemplate="<b>%{text}</b><br>Freq: %{x:.2f} Hz<br>Diss: %{y:.4f}<extra></extra>",
        )
    )

    # Mark Sounding Notes
    for note, freq in zip(selected_notes, sounding_notes):
        fig.add_vline(
            x=freq, line_width=1, line_dash="dash", line_color="white", opacity=0.5
        )
        fig.add_annotation(
            x=freq, y=max(dissonance_continuous), text=note, showarrow=False, yshift=10
        )

    fig.update_layout(
        title=f"Sensory Dissonance Curve (Context: {', '.join(selected_notes)})<br>Total Sounding Dissonance: {total_sounding_dissonance:.4f}",
        xaxis=dict(
            type="log", title="Frequency (Hz)", range=[np.log10(fmin), np.log10(fmax)]
        ),
        yaxis_title="Dissonance",
        template="plotly_dark",
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


@app.callback(
    Output("note-selector", "value"),
    Input("dissonance-graph", "clickData"),
    State("note-selector", "value"),
)
def update_selection_on_click(clickData, current_selection):
    if not clickData:
        return current_selection or []

    # Get the clicked point
    point = clickData["points"][0]

    # Check if we clicked on the "Piano Keys" trace
    # The curveNumber isn't always reliable if trace order changes, but we can check if 'text' is present and looks like a note
    # Our piano keys trace has 'text' populated with note names.
    clicked_note = point.get("text")

    if not clicked_note:
        return current_selection or []

    # Initialize selection if None
    if current_selection is None:
        current_selection = []

    # Toggle note
    if clicked_note in current_selection:
        current_selection.remove(clicked_note)
    else:
        current_selection.append(clicked_note)

    return current_selection


if __name__ == "__main__":
    app.run_server(debug=True)
