import os
import pickle
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import h5py

# ─────────────────────────────────────────────────────────────
# 1) define where your model .pkl’s live (display name → pickle path)
# ─────────────────────────────────────────────────────────────
MODEL_PKLS = {
    "BERT":    "results/bert-base-uncased_gdv/bert-base-uncased_bank_gdv.pkl",
    "DistilB": "results/distilbert-base-uncased_gdv/distilbert-base-uncased_bank_gdv.pkl",
    "GPT2":    "results/gpt2_gdv/gpt2_bank_gdv.pkl",
    "GPT-Neo": "results/EleutherAI/gpt-neo-1.3B_gdv/gpt-neo-1.3B_bank_gdv.pkl",
    "GPT-J":   "results/EleutherAI/gpt-j-6B_gdv/gpt-j-6B_bank_gdv.pkl",
}

# ─────────────────────────────────────────────────────────────
# 2) helper functions for GDV
# ─────────────────────────────────────────────────────────────
def compute_mean_intra_class_distance(X, labels, weights=None):
    vals = []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue
        dists, wts = [], []
        for i in idx:
            for j in idx:
                if i >= j:
                    continue
                d = np.linalg.norm(X[i] - X[j])
                w = (weights[i] * weights[j]) if weights is not None else 1.0
                dists.append(d * w)
                wts.append(w)
        if sum(wts) > 0:
            vals.append(sum(dists) / sum(wts))
    return float(np.mean(vals)) if vals else 0.0

def compute_mean_inter_class_distance(X, labels, weights=None):
    vals = []
    uniq = np.unique(labels)
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            idx1 = np.where(labels == uniq[i])[0]
            idx2 = np.where(labels == uniq[j])[0]
            if not len(idx1) or not len(idx2):
                continue
            dists, wts = [], []
            for ii in idx1:
                for jj in idx2:
                    d = np.linalg.norm(X[ii] - X[jj])
                    w = (weights[ii] * weights[jj]) if weights is not None else 1.0
                    dists.append(d * w)
                    wts.append(w)
            if sum(wts) > 0:
                vals.append(sum(dists) / sum(wts))
    return float(np.mean(vals)) if vals else 0.0

def compute_gdv(X, labels, weights=None):
    mu    = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    Xz = (X - mu) / sigma * 0.5
    if weights is None:
        weights = np.ones(Xz.shape[0], dtype=float)
    L = len(np.unique(labels))
    if L < 2:
        return 0.0
    intra = compute_mean_intra_class_distance(Xz, labels, weights)
    inter = compute_mean_inter_class_distance(Xz, labels, weights)
    D = Xz.shape[1]
    return float((1/np.sqrt(D)) * ((1/L) * intra - (2/(L * (L-1))) * inter))

# ─────────────────────────────────────────────────────────────
# 3) load static GDV pickles and set up HDF5 activation paths
# ─────────────────────────────────────────────────────────────
all_data = {}
for disp_name, pkl_path in MODEL_PKLS.items():
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing GDV data for {disp_name}: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # record best original GDV layer
    data['best_layer'] = min(
        data['gdv_per_layer'].keys(),
        key=lambda L: data['gdv_per_layer'][L]
    )
    # compute static axes from PCA coords
    coords = []
    for lyr in data['sorted_layers']:
        ld = data['layer_data'][lyr]
        coords += list(ld['x']) + list(ld['y'])
    arr = np.array(coords)
    data['static_min'] = float(arr.min()) if arr.size else -1.0
    data['static_max'] = float(arr.max()) if arr.size else 1.0
    # determine HDF5 directory for true activations
    model_id = data['meta']['model_name']  # e.g. 'distilbert-base-uncased'
    word = data['meta']['word']            # e.g. 'bank'
    h5_dir = os.path.join('results', 'activations', word, model_id.replace('/', '_'))
    data['h5_dir'] = h5_dir
    all_data[disp_name] = data

# ─────────────────────────────────────────────────────────────
# 4) build Dash panels
# ─────────────────────────────────────────────────────────────
panels = []
for disp_name, data in all_data.items():
    n_layers = len(data['sorted_layers'])
    panels.append(html.Div([
        html.H2(disp_name, style={'textAlign':'center'}),
        html.Div(id=f'info-{disp_name}', style={'textAlign':'center','margin':'0.5em'}),
        html.Div(id=f'neutral-info-{disp_name}', style={'textAlign':'center','margin':'0.5em'}),
        dcc.Store(id=f'ambig-{disp_name}', data=[]),
        dcc.Graph(
            id=f'graph-{disp_name}',
            config={
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['select2d','lasso2d'],
                'scrollZoom': True
            },
            style={'height':'400px'}
        ),
        html.Div([
            html.Label('Layer:'),
            dcc.Slider(
                id=f'slider-{disp_name}',
                min=1, max=n_layers, step=1, value=1,
                marks={i+1:str(i+1) for i in range(n_layers)},
                tooltip={'placement':'bottom'}
            )
        ], style={'marginTop':'1em','padding':'0 1em'})
    ], style={
        'border':'1px solid #ccc',
        'borderRadius':'8px',
        'padding':'1em',
        'background':'#fafafa'
    }))

# ─────────────────────────────────────────────────────────────
# 5) assemble Dash layout
# ─────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('GDV Dashboard', style={'textAlign':'center'}),
    html.Div(
        dcc.Checklist(
            id='axis-mode',
            options=[{'label':' Use static per-model axes','value':'static'}],
            value=[],
            labelStyle={'display':'inline-block','margin':'0 1em'}
        ),
        style={'textAlign':'center','marginBottom':'20px'}
    ),
    html.Div(panels,
        id='grid',
        style={
            'display':'grid','gridTemplateColumns':'repeat(3,1fr)',
            'gap':'1rem','padding':'1rem'
        }
    )
])

# ─────────────────────────────────────────────────────────────
# 6) callback to store ambiguous indices
# ─────────────────────────────────────────────────────────────
for disp_name in all_data:
    @app.callback(
        Output(f'ambig-{disp_name}','data'),
        Input(f'graph-{disp_name}','selectedData'),
        Input(f'graph-{disp_name}','clickData'),
        State(f'ambig-{disp_name}','data'),
        prevent_initial_call=True
    )
    def update_ambig(selectedData, clickData, ambig_idxs, disp_name=disp_name):
        triggered = callback_context.triggered[0]['prop_id'].split('.')[1]
        current = list(ambig_idxs or [])
        if triggered == 'selectedData':
            return [pt['pointNumber'] for pt in (selectedData or {}).get('points',[])]
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
# 7) main update: draw figure & recompute true neutralized GDV
# ─────────────────────────────────────────────────────────────
for disp_name, data in all_data.items():
    @app.callback(
        Output(f'graph-{disp_name}','figure'),
        Output(f'info-{disp_name}','children'),
        Output(f'neutral-info-{disp_name}','children'),
        Input(f'slider-{disp_name}','value'),
        Input('axis-mode','value'),
        Input(f'ambig-{disp_name}','data'),
        prevent_initial_call=False
    )
    def update_panel(selected_layer, axis_mode, ambig_idxs, disp_name=disp_name, data=data):
        # 2D coords
        layer_idx = selected_layer - 1
        lyr = data['sorted_layers'][layer_idx]
        ld = data['layer_data'][lyr]
        x = np.array(ld['x']); y = np.array(ld['y'])
        grp = np.array(ld['group']); sents = np.array(ld['sentence'])

        # load full activations
        h5_path = os.path.join(data['h5_dir'], f'layer_{lyr}.h5')
        with h5py.File(h5_path,'r') as f:
            X_full = f['X'][:]
            labels_full = f['labels'][:]
            mu = f['mu'][:]; sigma = f['sigma'][:]
        # weights
        w = np.ones(len(labels_full))
        if ambig_idxs:
            w[ambig_idxs] = 0.5
        # GDV computations
        orig_gdv = compute_gdv(X_full, labels_full)
        true_neut = compute_gdv(X_full, labels_full, weights=w)

        # build scatter traces
        fig = go.Figure()
        colors = ['blue','red','green','orange','purple','brown']
        for g in np.unique(grp):
            mask = grp==g
            inds = np.where(mask)[0]
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask], mode='markers',
                marker=dict(
                    size=10, color=colors[g%len(colors)],
                    line=dict(
                        width=[2 if i in ambig_idxs else 0 for i in inds],
                        color='black'
                    )
                ),
                name=f'Group {g}', customdata=sents[mask],
                hovertemplate='%{customdata}<extra></extra>'
            ))
        # overlay ambiguous
        if ambig_idxs:
            fig.add_trace(go.Scatter(
                x=x[ambig_idxs], y=y[ambig_idxs], mode='markers',
                marker=dict(size=14, color='yellow', line=dict(color='black',width=1)),
                name='Ambiguous', hoverinfo='skip'
            ))
        # layout
        fig.update_layout(
            title=f'Layer {selected_layer}: orig GDV={orig_gdv:.4f}',
            xaxis_title='PC 1', yaxis_title='PC 2', legend_title='Group',
            clickmode='event+select'
        )
        if 'static' in axis_mode:
            fig.update_xaxes(range=[data['static_min'], data['static_max']])
            fig.update_yaxes(range=[data['static_min'], data['static_max']])

        # info lines
        best_orig = data['best_layer'] + 1
        info = (
            f"Word='{data['meta']['word']}'  "
            f"Layers=1–{len(data['sorted_layers'])}  "
            f"Best GDV Layer={best_orig}"
        )
        if not ambig_idxs:
            neutral_info = f"No ambiguous selected; neutralized GDV = {orig_gdv:.4f}"
        else:
            neutral_info = f"True neutralized GDV = {true_neut:.4f}" 
        return fig, info, neutral_info

# ─────────────────────────────────────────────────────────────
# 8) run server
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=True)