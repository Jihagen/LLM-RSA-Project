import os
import re
import torch
from torch.cuda.amp import autocast
import pandas as pd
from dotenv import load_dotenv
from models import load_model_and_tokenizer, get_target_activations
from data import flatten_dataframe
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import csv
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import pickle

# ---- GDV Helper Functions ----
def compute_mean_intra_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    intra_dists = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            continue
        dists = pdist(X[idx], metric="euclidean")
        intra_dists.append(np.mean(dists))
    if len(intra_dists) == 0:
        return 0.0
    return np.mean(intra_dists)

def compute_mean_inter_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            idx1 = np.where(labels == unique_labels[i])[0]
            idx2 = np.where(labels == unique_labels[j])[0]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            diff = X[idx1][:, None, :] - X[idx2]
            dists = np.linalg.norm(diff, axis=2)
            inter_dists.append(np.mean(dists))
    if len(inter_dists) == 0:
        return 0.0
    return np.mean(inter_dists)

def compute_gdv(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    L = len(unique_labels)
    if L < 2:
        return 0.0
    intra = compute_mean_intra_class_distance(X, labels)
    inter = compute_mean_inter_class_distance(X, labels)
    D = X.shape[1]
    gdv = (1/np.sqrt(D)) * ((1/L) * intra - (2/(L * (L - 1))) * inter)
    return gdv

# ---- Plotting Function (unchanged) ----
def plot_layer_activations(activations: np.ndarray, labels: np.ndarray, layer_idx: int, gdv_value: float, output_dir='results/gdv/plots'):
    pca = PCA(n_components=2)
    activations_2d = pca.fit_transform(activations)
    
    plt.figure(figsize=(6, 6))
    
    unique_groups = np.unique(labels)
    for group in unique_groups:
        group_mask = labels == group
        plt.scatter(activations_2d[group_mask, 0],
                    activations_2d[group_mask, 1],
                    label=f'Group {group}')
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Layer {layer_idx}\nGDV = {gdv_value:.4f}")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"layer_{layer_idx}_activations.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

# ---- Animation Function (unchanged) ----
def animate_layers_smooth(activations_dict: dict, semantic_labels: np.ndarray, hold_count: int = 3, interp_count: int = 5, interval: int = 500):
    sorted_layers = sorted(activations_dict.keys())
    pca_data = {}
    gdv_data = {}
    for layer in sorted_layers:
        X = activations_dict[layer].numpy()
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        pca_data[layer] = X_2d
        gdv_data[layer] = compute_gdv(X, semantic_labels)
    
    schedule = []
    L = len(sorted_layers)
    for i in range(L - 1):
        for _ in range(hold_count):
            schedule.append(("recorded", i, 0))
        for j in range(1, interp_count):
            frac = j / (interp_count - 1)
            schedule.append(("interp", i, frac))
    for _ in range(hold_count):
        schedule.append(("recorded", L - 1, 0))
    
    total_frames = len(schedule)
    
    unique_groups = np.unique(semantic_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_groups))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    scatters = {}
    for idx, group in enumerate(unique_groups):
        scat = ax.scatter([], [], color=colors(idx), label=f"Group {group}")
        scatters[group] = scat
    title = ax.set_title("")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    
    all_data = np.concatenate(list(pca_data.values()), axis=0)
    x_min, x_max = all_data[:,0].min(), all_data[:,0].max()
    y_min, y_max = all_data[:,1].min(), all_data[:,1].max()
    ax.set_xlim(x_min - 0.1*(x_max - x_min), x_max + 0.1*(x_max - x_min))
    ax.set_ylim(y_min - 0.1*(y_max - y_min), y_max + 0.1*(y_max - y_min))
    
    def init():
        for group in unique_groups:
            scatters[group].set_offsets(np.empty((0, 2)))
        title.set_text("")
        return list(scatters.values()) + [title]
    
    def update(frame):
        mode, base_idx, frac = schedule[frame]
        if mode == "recorded":
            cur_layer = sorted_layers[base_idx]
            data_2d = pca_data[cur_layer]
            cur_gdv = gdv_data[cur_layer]
            title_text = f"Layer {cur_layer}: GDV = {cur_gdv:.4f}"
        elif mode == "interp":
            layer_a = sorted_layers[base_idx]
            layer_b = sorted_layers[base_idx+1]
            data_2d = (1-frac)*pca_data[layer_a] + frac*pca_data[layer_b]
            cur_gdv = (1-frac)*gdv_data[layer_a] + frac*gdv_data[layer_b]
            title_text = f"Transition: {layer_a}→{layer_b} (t={frac:.2f})"
        else:
            data_2d = None
            title_text = ""
        for group in unique_groups:
            mask = semantic_labels == group
            points = data_2d[mask]
            scatters[group].set_offsets(points)
        title.set_text(title_text)
        return list(scatters.values()) + [title]
    
    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init,
                                  interval=interval, blit=False, repeat=True)
    plt.show()
    return ani

# ---- Main Function to Run GDV Experiment and Save Data for Dash ----

def run_gdv_experiment(df, model_name):
    # Flatten the synthetic data.
    df_flat = flatten_dataframe(df)
    print("Flattened dataset:")
    print(df_flat.head())
    
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    sentences = df_flat["sentence"].tolist()
    target_words = df_flat["word"].tolist()
    semantic_labels = np.array(df_flat["semantic_group_id"].tolist())
    
    # Get token-level activations.
    activations = get_target_activations(model, tokenizer, sentences, target_words, batch_size=4)
    
    for layer_idx, act_tensor in activations.items():
        print(f"Layer {layer_idx} activation shape: {act_tensor.shape}")
    
    # Compute GDV per layer and produce plots.
    gdv_all = {}
    pca_data = {}  # we store PCA projections per layer for saving
    sorted_layers = sorted(activations.keys())
    for layer_idx in sorted_layers:
        act_tensor = activations[layer_idx]  # shape: [N, hidden_dim]
        X = act_tensor.numpy()
        gdv_value = compute_gdv(X, semantic_labels)
        gdv_all[layer_idx] = gdv_value
        print(f"Layer {layer_idx}: GDV = {gdv_value:.4f}")
        # Save the PCA projection used for plotting.
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        pca_data[layer_idx] = X_2d
        plot_layer_activations(X, semantic_labels, layer_idx, gdv_value)
    
    # Save GDV CSV as before.
    output_dir = 'results/gdv/'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "gdv_values.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Layer", "GDV"])
        for layer_idx, gdv in gdv_all.items():
            writer.writerow([layer_idx, gdv])
    print(f"GDV values saved to {csv_path}")
    
    # (Optional) Run the smooth animation.
    ani = animate_layers_smooth(activations, semantic_labels, hold_count=3, interp_count=5, interval=500)
    
    # Build metadata for Dash.
    max_gdv_layer = max(gdv_all, key=lambda k: gdv_all[k])
    meta = {
        "model_name": model_name,
        "word": target_words[0],
        "total_layers": len(sorted_layers),
        "max_gdv_layer": max_gdv_layer
    }
    
    # Build the output dictionary to save for the Dash app.
    # For each layer we store:
    #    - x: PCA x coordinates (from our precomputed pca_data),
    #    - y: PCA y coordinates,
    #    - group: the semantic labels (for each sample),
    #    - sentence: the sample sentences.
    layer_info = {}
    for layer in sorted_layers:
        X_2d = pca_data[layer]
        layer_info[layer] = {
            "x": X_2d[:, 0],
            "y": X_2d[:, 1],
            "group": semantic_labels,  # 1D array, same for all layers
            "sentence": sentences      # order should match activations; one sentence per sample
        }
    
    output_data = {
        "sorted_layers": sorted_layers,
        "layer_data": layer_info,
        "gdv_per_layer": gdv_all,
        "meta": meta
    }
    
    # Save the output dictionary.
    output_filename = f"{model_name}_{target_words[0]}_gdv.pkl"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)
    print(f"Saved GDV and layer data to {output_path}")
    
    return gdv_all


