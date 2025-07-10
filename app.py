import os
import pickle
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.distance import pdist

# ─────────────────────────────────────────────────────────────
# 1) define where your model .pkl’s live
# ─────────────────────────────────────────────────────────────
MODEL_PKLS = {
    "BERT":    "results/bert-base-uncased_gdv/bert-base-uncased_bank_gdv.pkl",
    "DistilB": "results/distilbert-base-uncased_gdv/distilbert-base-uncased_bank_gdv.pkl",
    "GPT2":    "results/gpt2_gdv/gpt2_bank_gdv.pkl",
    "GPT-Neo": "results/EleutherAI/gpt-neo-1.3B_gdv/gpt-neo-1.3B_bank_gdv.pkl",
    "GPT-J":   "results/EleutherAI/gpt-j-6B_gdv/gpt-j-6B_bank_gdv.pkl",
}

# ─────────────────────────────────────────────────────────────
# 2) helper functions for GDV (on any X, labels) ─────────────
# ─────────────────────────────────────────────────────────────
def compute_mean_intra_class_distance(X, labels, weights=None):
    vals = []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2: continue
        dists, wts = [], []
        for i in idx:
            for j in idx:
                if i >= j: continue
                d = np.linalg.norm(X[i] - X[j])
                w = (weights[i]*weights[j]) if weights is not None else 1.0
                dists.append(d*w)
                wts.append(w)
        if sum(wts) > 0:
            vals.append(sum(dists)/sum(wts))
    return float(np.mean(vals)) if vals else 0.0


def compute_mean_inter_class_distance(X, labels, weights=None):
    vals = []
    uniq = np.unique(labels)
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            idx1 = np.where(labels==uniq[i])[0]
            idx2 = np.where(labels==uniq[j])[0]
            if not len(idx1) or not len(idx2): continue
            dists, wts = [], []
            for ii in idx1:
                for jj in idx2:
                    d = np.linalg.norm(X[ii] - X[jj])
                    w = (weights[ii]*weights[jj]) if weights is not None else 1.0
                    dists.append(d*w)
                    wts.append(w)
            if sum(wts) > 0:
                vals.append(sum(dists)/sum(wts))
    return float(np.mean(vals)) if vals else 0.0


def compute_gdv(X, labels, weights=None):
    # z-score + scale
    mu, sigma = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-12
    Xz = (X - mu)/sigma * 0.5
    D, L = Xz.shape[1], len(np.unique(labels))
    if L < 2: return 0.0
    intra = compute_mean_intra_class_distance(Xz, labels, weights)
    inter = compute_mean_inter_class_distance(Xz, labels, weights)
    return float((1/np.sqrt(D))*((1/L)*intra - (2/(L*(L-1)))*inter))

# ─────────────────────────────────────────────────────────────
# 3) load data and compute static axes
# ─────────────────────────────────────────────────────────────
all_data = {}
for name, path in MODEL_PKLS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing GDV data for {name}: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    data["best_layer"] = min(
        data["gdv_per_layer"].keys(),
        key=lambda L: data["gdv_per_layer"][L]
    )
    # static axes from PCA coords
    coords = []
    for layer in data["sorted_layers"]:
        ld = data["layer_data"][layer]
        coords += list(ld["x"]) + list(ld["y"])
    arr = np.array(coords)
    data["static_min"] = float(arr.min()) if arr.size else -1.0
    data["static_max"] = float(arr.max()) if arr.size else 1.0
    all_data[name] = data

# ─────────────────────────────────────────────────────────────
# 4) build Dash panels
# ─────────────────────────────────────────────────────────────
panels = []
for name, data in all_data.items():
    n_layers = len(data["sorted_layers"])
    panels.append(html.Div([
        html.H2(name, style={"textAlign":"center"}),
        html.Div(id=f"info-{name}", style={"textAlign":"center","margin":"0.5em"}),
        html.Div(id=f"neutral-info-{name}", style={"textAlign":"center","margin":"0.5em"}),
        dcc.Store(id=f"ambig-{name}", data=[]),
        dcc.Graph(
            id=f"graph-{name}",
            config={
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["select2d","lasso2d"],
                "scrollZoom": True
            },
            style={"height":"400px"}
        ),
        html.Div([
            html.Label("Layer:"),
            dcc.Slider(
                id=f"slider-{name}",
                min=1, max=n_layers, step=1, value=1,
                marks={i+1:str(i+1) for i in range(n_layers)},
                tooltip={"placement":"bottom"}
            )
        ], style={"marginTop":"1em","padding":"0 1em"})
    ], style={
        "border":"1px solid #ccc",
        "borderRadius":"8px",
        "padding":"1em",
        "background":"#fafafa"
    }))

# ─────────────────────────────────────────────────────────────
# 5) Dash layout
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
    html.Div(panels, id="grid", style={
        "display":"grid","gridTemplateColumns":"repeat(3,1fr)",
        "gap":"1rem","padding":"1rem"
    })
])

# ─────────────────────────────────────────────────────────────
# 6) callback: update ambiguous indices store
#    supports box/lasso (selectedData) and single-click (clickData)
# ─────────────────────────────────────────────────────────────
for name in all_data:
    @app.callback(
        Output(f"ambig-{name}", "data"),
        Input(f"graph-{name}", "selectedData"),
        Input(f"graph-{name}", "clickData"),
        State(f"ambig-{name}", "data"),
        prevent_initial_call=True
    )
    def update_ambig(selectedData, clickData, ambig_idxs, name=name):
        triggered = callback_context.triggered[0]["prop_id"].split('.')[1]
        current = list(ambig_idxs or [])
        if triggered == 'selectedData':
            pts = selectedData.get('points', []) if selectedData else []
            return [pt['pointNumber'] for pt in pts]
        if triggered == 'clickData':
            if not clickData or 'points' not in clickData:
                return current
            idx = clickData['points'][0]['pointNumber']
            if idx in current:
                current.remove(idx)
            else:
                current.append(idx)
            return current
        return ambig_idxs

# ─────────────────────────────────────────────────────────────
# 7) main update: draw figure & compute GDVs
# ─────────────────────────────────────────────────────────────
for name, data in all_data.items():
    @app.callback(
        Output(f"graph-{name}", "figure"),
        Output(f"info-{name}", "children"),
        Output(f"neutral-info-{name}", "children"),
        Input(f"slider-{name}", "value"),
        Input("axis-mode", "value"),
        Input(f"ambig-{name}", "data"),
        prevent_initial_call=False
    )
    def update_panel(selected_layer, axis_mode, ambig_idxs, name=name, data=data):
        # prepare layer data
        layer_idx = selected_layer - 1
        layer = data['sorted_layers'][layer_idx]
        ld = data['layer_data'][layer]
        x = np.array(ld['x']); y = np.array(ld['y'])
        grp = np.array(ld['group']); sents = np.array(ld['sentence'])
        X2d = np.stack([x, y], axis=1)

        # compute original & current-neutralized for this layer
        orig_gdv = compute_gdv(X2d, grp, weights=None)
        weights = np.ones(len(grp))
        if ambig_idxs:
            weights[ambig_idxs] = 0.5
        curr_neut = compute_gdv(X2d, grp, weights=weights)

        # compute best neutralized GDV across all layers
        best_neut_val = None
        best_neut_layer = None
        if ambig_idxs:
            neut_vals = {}
            for lyr in data['sorted_layers']:
                ld2 = data['layer_data'][lyr]
                X2 = np.stack([ld2['x'], ld2['y']], axis=1)
                neut_vals[lyr] = compute_gdv(X2, grp, weights=weights)
            best_neut_layer = min(neut_vals, key=lambda L: neut_vals[L])
            best_neut_val = neut_vals[best_neut_layer]

        # build scatter traces
        fig = go.Figure()
        colors = ['blue','red','green','orange','purple','brown']
        for g in np.unique(grp):
            mask = grp == g
            inds = np.where(mask)[0]
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask], mode='markers',
                marker=dict(
                    size=10, color=colors[g % len(colors)],
                    line=dict(
                        width=[2 if i in ambig_idxs else 0 for i in inds],
                        color='black'
                    )
                ),
                name=f'Group {g}', customdata=sents[mask],
                hovertemplate='%{customdata}<extra></extra>'
            ))
        # overlay ambiguous points
        if ambig_idxs:
            fig.add_trace(go.Scatter(
                x=x[ambig_idxs], y=y[ambig_idxs], mode='markers',
                marker=dict(size=14, color='yellow', line=dict(color='black', width=1)),
                name='Ambiguous', hoverinfo='skip'
            ))

        # layout & axes
        fig.update_layout(
            title=f"Layer {selected_layer}: orig GDV={orig_gdv:.4f}",
            xaxis_title='PC 1', yaxis_title='PC 2', legend_title='Group',
            clickmode='event+select'
        )
        if 'static' in axis_mode:
            fig.update_xaxes(range=[data['static_min'], data['static_max']])
            fig.update_yaxes(range=[data['static_min'], data['static_max']])

        # info text
        orig_best = data['best_layer'] + 1
        info = f"Word='{data['meta']['word']}'  Layers=1–{len(data['sorted_layers'])}  " \
               f"Best GDV Layer={orig_best} ({data['gdv_per_layer'][data['best_layer']]:.4f})"
        if not ambig_idxs:
            neutral_info = f"No ambiguous selected; neutralized GDV (current) = {orig_gdv:.4f}"
        else:
            neutral_info = (
                f"Current neutralized GDV = {curr_neut:.4f}; "
                f"Best neutralized Layer={best_neut_layer+1} ({best_neut_val:.4f})"
            )

        return fig, info, neutral_info

# ─────────────────────────────────────────────────────────────
# 8) run server
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=True)
