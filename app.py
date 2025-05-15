import os
import pickle
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1) define where your model .pkl’s live
# ─────────────────────────────────────────────────────────────
MODEL_PKLS = {
    "BERT":     "results/bert-base-uncased_gdv/bert-base-uncased_bank_gdv.pkl",
    "DistilB":  "results/distilbert-base-uncased_gdv/distilbert-base-uncased_bank_gdv.pkl",
    "GPT2":     "results/gpt2_gdv/gpt2_bank_gdv.pkl",
    "GPT-Neo":  "results/EleutherAI/gpt-neo-1.3B_gdv/gpt-neo-1.3B_bank_gdv.pkl",
    "GPT-J":    "results/EleutherAI/gpt-j-6B_gdv/gpt-j-6B_bank_gdv.pkl",
}

# ─────────────────────────────────────────────────────────────
# 2) load each, compute per-model static axes
# ─────────────────────────────────────────────────────────────
all_data = {}
for name, pth in MODEL_PKLS.items():
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Missing GDV data for {name}: {pth}")
    with open(pth, "rb") as f:
        data = pickle.load(f)
        
    # find the best layer by GDV
    data["best_layer"] = min(
        data["gdv_per_layer"].keys(),
        key=lambda L: data["gdv_per_layer"][L]
    )

    # collect every layer's PCA coords into one big list
    coords = []
    for layer in data["sorted_layers"]:
        ld = data["layer_data"][layer]
        coords.extend(ld["x"])
        coords.extend(ld["y"])

    arr = np.array(coords)
    if arr.size:
        data["static_min"] = float(arr.min())
        data["static_max"] = float(arr.max())
    else:
        data["static_min"], data["static_max"] = -1.0, 1.0

    all_data[name] = data

# ─────────────────────────────────────────────────────────────
# 3) build a list of panels for the grid
# ─────────────────────────────────────────────────────────────
panels = []
for name, data in all_data.items():
    n_layers = len(data["sorted_layers"])
    panel = html.Div([
        html.H2(name, style={"textAlign":"center"}),
        html.Div(id=f"info-{name}", style={"textAlign":"center","margin":"0.5em"}),
        dcc.Graph(id=f"graph-{name}"),
        html.Div([
            html.Label("Layer:"),
            dcc.Slider(
                id=f"slider-{name}",
                min=1,
                max=n_layers,
                step=1,
                value=1,
                marks={i+1: str(i+1) for i in range(n_layers)},
                tooltip={"placement":"bottom"}
            )
        ], style={"marginTop":"1em","padding":"0 1em"})
    ], style={
        "border":"1px solid #ccc",
        "borderRadius":"8px",
        "padding":"1em",
        "background":"#fafafa"
    })
    panels.append(panel)

# ─────────────────────────────────────────────────────────────
# 4) assemble the overall layout with a 3×2 grid container
# ─────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("GDV Dashboard", style={"textAlign":"center"}),
    html.Div(
        dcc.Checklist(
            id="axis-mode",
            options=[{"label":" Use static per-model axes","value":"static"}],
            value=[],
            labelStyle={"display":"inline-block","margin":"0 1em"}
        ),
        style={"textAlign":"center","marginBottom":"20px"}
    ),
    html.Div(
        panels,
        id="grid",
        style={
            "display":"grid",
            "gridTemplateColumns":"repeat(3,1fr)",
            "gap":"1rem",
            "padding":"1rem"
        }
    )
])

# ─────────────────────────────────────────────────────────────
# 5) one callback per model panel
# ─────────────────────────────────────────────────────────────
for name, data in all_data.items():
    @app.callback(
        Output(f"graph-{name}", "figure"),
        Output(f"info-{name}", "children"),
        Input(f"slider-{name}", "value"),
        Input("axis-mode", "value"),
        prevent_initial_call=False
    )
    def update_panel(selected_1based, axis_mode, name=name, data=data):
        # map 1-based slider → 0-based layer index
        idx0 = selected_1based - 1
        layer = data["sorted_layers"][idx0]
        ld = data["layer_data"][layer]

        x   = np.array(ld["x"])
        y   = np.array(ld["y"])
        grp = np.array(ld["group"])
        sents = np.array(ld["sentence"])

        # build traces
        fig = go.Figure()
        colors = ["blue","red","green","orange","purple","brown"]
        for g in np.unique(grp):
            mask = (grp == g)
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask],
                mode="markers",
                marker=dict(size=10, color=colors[g % len(colors)]),
                name=f"Group {g}",
                customdata=sents[mask],
                hovertemplate="%{customdata}<extra></extra>"
            ))

        fig.update_layout(
            title=f"Layer {selected_1based}: GDV = {data['gdv_per_layer'][layer]:.4f}",
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            legend_title="Group"
        )

        # apply static axes if requested
        if "static" in axis_mode:
            fig.update_xaxes(range=[data["static_min"], data["static_max"]])
            fig.update_yaxes(range=[data["static_min"], data["static_max"]])

        # info line
        best1 = data["best_layer"]
        info = (
            f"Word='{data['meta']['word']}'  "
            f"Layers=1–{len(data['sorted_layers'])}  "
            f"Best GDV Layer={best1+1}"
        )
        return fig, info

# ─────────────────────────────────────────────────────────────
# 6) run the server
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run_server(debug=True)
