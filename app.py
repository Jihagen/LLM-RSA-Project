import os
import pickle
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import h5py

# ─────────────────────────────────────────────────────────────
# 1) Define your words and models
# ─────────────────────────────────────────────────────────────
WORDS = ["bank", "bat", "light", "spring", "too"]
MODELS = {
    "BERT":    "bert-base-uncased",
    "DistilB": "distilbert-base-uncased",
    "GPT2":    "gpt2",
    "GPT-Neo": "EleutherAI_gpt-neo-1.3B",
}

# ─────────────────────────────────────────────────────────────
# 2) Helper functions for GDV
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
# 3) Load static GDV pickles and set up HDF5 activation paths
# ─────────────────────────────────────────────────────────────
all_data = {}
for word in WORDS:
    all_data[word] = {}
    for disp_name, model_id in MODELS.items():
        # path to your pickle
        pkl_path = os.path.join("results", word, f"{model_id}_gdv", f"{model_id}_{word}_dash.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing GDV data for {word}/{disp_name}: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # record best original GDV layer
        data['best_layer'] = min(
            data['gdv_per_layer'].keys(),
            key=lambda L: data['gdv_per_layer'][L]
        )
        # compute static PCA axes
        coords = []
        for lyr in data['sorted_layers']:
            ld = data['layer_data'][lyr]
            coords += list(ld['x']) + list(ld['y'])
        arr = np.array(coords)
        data['static_min'] = float(arr.min()) if arr.size else -1.0
        data['static_max'] = float(arr.max()) if arr.size else 1.0
        # HDF5 activation directory
        safe_id = model_id.replace('/', '_')
        data['h5_dir'] = os.path.join('results', 'activations', word, safe_id)

        all_data[word][disp_name] = data

# ─────────────────────────────────────────────────────────────
# 4) Build Dash layout
# ─────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('GDV Dashboard', style={'textAlign':'center'}),

    # word selector dropdown
    html.Div(
        dcc.Dropdown(
            id='word-selector',
            options=[{'label': w, 'value': w} for w in WORDS],
            value=WORDS[0],
            clearable=False,
            style={'width':'180px','margin':'0 auto'}
        ),
        style={'textAlign':'center','margin':'1em'}
    ),

    # static axes toggle
    html.Div(
        dcc.Checklist(
            id='axis-mode',
            options=[{'label':' Use static per-model axes','value':'static'}],
            value=[],
            labelStyle={'display':'inline-block','margin':'0 1em'}
        ),
        style={'textAlign':'center','marginBottom':'20px'}
    ),

    # container for panels
    html.Div(
        id='panels-container',
        style={
            'display':'grid',
            'gap':'1rem','padding':'1rem'
        },
        children=[
            # one Div per word, hide/show via callback
            html.Div(
                id=f'panels-{word}',
                style={ 'display': 'grid',
                'gridTemplateColumns': 'repeat(2, 1fr)',
                'gap': '1rem',
                'border': '1px solid #ccc',
                'borderRadius': '8px',
                'padding': '1em',
                'background': '#fafafa'
                },
                children=[
                    # one panel per model for this word
                    html.Div([
                        html.H2(disp_name, style={'textAlign':'center'}),
                        html.Div(id=f'info-{word}-{disp_name}',
                                 style={'textAlign':'center','margin':'0.5em'}),
                        html.Div(id=f'neutral-{word}-{disp_name}',
                                 style={'textAlign':'center','margin':'0.5em'}),
                        dcc.Store(id=f'ambig-{word}-{disp_name}', data=[]),
                        dcc.Graph(id=f'graph-{word}-{disp_name}',
                                  config={
                                      'displayModeBar':True,
                                      'modeBarButtonsToAdd':['select2d','lasso2d'],
                                      'scrollZoom':True
                                  }, style={'height':'350px'}),
                        html.Div([
                            html.Label('Layer:'),
                            dcc.Slider(
                                id=f'slider-{word}-{disp_name}',
                                min=1,
                                max=len(all_data[word][disp_name]['sorted_layers']),
                                step=1,
                                value=1,
                                marks={i+1:str(i+1)
                                       for i in range(len(all_data[word][disp_name]['sorted_layers']))},
                                tooltip={'placement':'bottom'}
                            )
                        ], style={'marginTop':'1em','padding':'0 1em'})
                    ], style={'marginBottom':'2em'})
                    for disp_name in MODELS
                ]
            )
            for i, word in enumerate(WORDS)
        ]
    )
])

# ─────────────────────────────────────────────────────────────
# 5) Callback: show only selected word’s panels
# ─────────────────────────────────────────────────────────────
@app.callback(
    [Output(f'panels-{w}', 'style') for w in WORDS],
    Input('word-selector', 'value')
)
def swap_word(selected):
    base = { 'display': 'grid',
    'gridTemplateColumns': 'repeat(2, 1fr)',
    'gap': '1rem',
    'border': '1px solid #ccc',
    'borderRadius': '8px',
    'padding': '1em',
    'background': '#fafafa'}
    return [
        {**base, 'display':'grid'} if w==selected else {**base,'display':'none'}
        for w in WORDS
    ]

# ─────────────────────────────────────────────────────────────
# 6) Callbacks for ambiguous selection and figure updates
# ─────────────────────────────────────────────────────────────
for word in WORDS:
    for disp_name in MODELS:
        # track ambiguous indices
        @app.callback(
            Output(f'ambig-{word}-{disp_name}', 'data'),
            Input(f'graph-{word}-{disp_name}', 'selectedData'),
            Input(f'graph-{word}-{disp_name}', 'clickData'),
            State(f'ambig-{word}-{disp_name}', 'data'),
            prevent_initial_call=True
        )
        def update_ambig(selectedData, clickData, ambig_idxs,
                         word=word, disp_name=disp_name):
            triggered = callback_context.triggered[0]['prop_id'].split('.')[1]
            current = list(ambig_idxs or [])
            if triggered == 'selectedData':
                return [pt['pointNumber'] for pt in (selectedData or {}).get('points', [])]
            if triggered == 'clickData' and clickData and 'points' in clickData:
                idx = clickData['points'][0]['pointNumber']
                if idx in current:
                    current.remove(idx)
                else:
                    current.append(idx)
                return current
            return ambig_idxs

        # update graph & info
        @app.callback(
            Output(f'graph-{word}-{disp_name}', 'figure'),
            Output(f'info-{word}-{disp_name}', 'children'),
            Output(f'neutral-{word}-{disp_name}', 'children'),
            Input(f'slider-{word}-{disp_name}', 'value'),
            Input('axis-mode', 'value'),
            Input(f'ambig-{word}-{disp_name}', 'data'),
            prevent_initial_call=False
        )
        def update_panel(selected_layer, axis_mode, ambig_idxs,
                         word=word, disp_name=disp_name):
            data = all_data[word][disp_name]
            lyr = data['sorted_layers'][selected_layer - 1]
            ld = data['layer_data'][lyr]
            x, y = np.array(ld['x']), np.array(ld['y'])
            grp = np.array(ld['group']); sents = np.array(ld['sentence'])

            # load full activations
            h5_dir = data['h5_dir']
            with h5py.File(os.path.join(h5_dir, f'layer_{lyr}.h5'), 'r') as f:
                X_full = f['X'][:]
                labels_full = f['labels'][:]

            # compute GDV
            wts = np.ones(len(labels_full))
            if ambig_idxs:
                wts[ambig_idxs] = 0.5
            orig = compute_gdv(X_full, labels_full)
            neut = compute_gdv(X_full, labels_full, weights=wts)

            # build figure
            fig = go.Figure()
            colors = ['blue','red','green','orange','purple','brown']
            for g in np.unique(grp):
                mask = grp == g
                idxs = np.where(mask)[0]
                fig.add_trace(go.Scatter(
                    x=x[mask], y=y[mask], mode='markers',
                    marker=dict(
                        size=10,
                        color=colors[int(g) % len(colors)],
                        line=dict(
                            width=[2 if i in (ambig_idxs or []) else 0 for i in idxs],
                            color='black'
                        )
                    ),
                    name=f'Group {g}',
                    customdata=sents[mask],
                    hovertemplate='%{customdata}<extra></extra>'
                ))
            if ambig_idxs:
                fig.add_trace(go.Scatter(
                    x=x[ambig_idxs], y=y[ambig_idxs], mode='markers',
                    marker=dict(size=14, color='yellow', line=dict(color='black', width=1)),
                    name='Ambiguous', hoverinfo='skip'
                ))

            fig.update_layout(
                title=f'Layer {selected_layer}: orig GDV={orig:.4f}',
                xaxis_title='PC 1', yaxis_title='PC 2', legend_title='Group',
                clickmode='event+select'
            )
            if 'static' in axis_mode:
                fig.update_xaxes(range=[data['static_min'], data['static_max']])
                fig.update_yaxes(range=[data['static_min'], data['static_max']])

            info_txt = (
                f"Word='{word}'  Layers=1–{len(data['sorted_layers'])}  "
                f"Best GDV Layer={data['best_layer']+1}"
            )
            neutral_txt = (
                f"No ambiguous selected; neutralized GDV = {orig:.4f}" if not ambig_idxs
                else f"True neutralized GDV = {neut:.4f}"
            )
            return fig, info_txt, neutral_txt

# ─────────────────────────────────────────────────────────────
# 7) Run server
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=True)
