import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import pickle
import os

# Path to the pre-saved GDV data file.
pkl_path = "results/gdv/bert-base-uncased_bank_gdv.pkl"

if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"File {pkl_path} not found. Please run your GDV experiment script first.")

# Load the saved output dictionary.
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# Extract the necessary pieces.
sorted_layers = data["sorted_layers"]           # e.g. [0, 1, 2, ..., 10]
layer_data = data["layer_data"]                   # dict mapping layer index -> dict with keys "x", "y", "group", "sentence"
gdv_per_layer = data["gdv_per_layer"]             # dict mapping layer index -> GDV (float)
meta = data["meta"]     
best_layer = min(gdv_per_layer, key=lambda k: gdv_per_layer[k])
meta["best_layer"] = best_layer                          # dict with keys "model_name", "word", "total_layers", "max_gdv_layer"

# Create the Dash app.
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("GDV Visualization Dashboard"),
    html.Div(id='model-info', style={'fontSize': '18px', 'marginBottom': '20px'}),
    dcc.Graph(id='gdv-graph'),
    html.Div([
        html.Label("Select Layer:"),
        dcc.Slider(
            id="layer-slider",
            min=sorted_layers[0],
            max=sorted_layers[-1],
            step=1,
            value=sorted_layers[0],
            marks={str(layer): str(layer) for layer in sorted_layers}
        )
    ], style={'width': '80%', 'margin': 'auto'})
])

@app.callback(
    [Output('gdv-graph', 'figure'),
     Output('model-info', 'children')],
    [Input('layer-slider', 'value')]
)
def update_graph(selected_layer):
    # Retrieve the stored PCA data and sample info for the selected layer.
    current_data = layer_data[selected_layer]
    x_vals = np.array(current_data["x"])  # 1D array of PCA x coordinates
    y_vals = np.array(current_data["y"])  # 1D array of PCA y coordinates
    groups = np.array(current_data["group"])  # array of semantic group labels (length = number of samples)
    sentences = np.array(current_data["sentence"])  # sample sentences for hover text
    
    # Build the figure.
    fig = go.Figure()
    
    # Assuming two groups; you can extend the color mapping as needed.
    color_map = {0: "blue", 1: "red"}
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group  # boolean mask of same length as x_vals
        fig.add_trace(go.Scatter(
            x=x_vals[mask],
            y=y_vals[mask],
            mode="markers",
            marker=dict(color=color_map.get(group, "gray"), size=12),
            name=f"Group {group}",
            customdata=sentences[mask],
            hovertemplate="%{customdata}<extra></extra>"
        ))
    
    # Update layout with current layer and its GDV.
    fig.update_layout(
        title=f"Layer {selected_layer}: GDV = {gdv_per_layer[selected_layer]:.4f}",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title="Semantic Group"
    )
    
    # Build the model info display.
    model_info_text = (
        f"Model: {meta['model_name']} | Word: '{meta['word']}' | Total Layers: {meta['total_layers']} | "
        f"Highest GDV Layer: {meta['best_layer']}"
    )
    
    return fig, model_info_text

if __name__ == "__main__":
    app.run_server(debug=True)
