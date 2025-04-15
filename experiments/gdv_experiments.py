
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
import torch
import csv
from sklearn.decomposition import PCA
import matplotlib.animation as animation

def run_gdv_experiment(df, model_name):
    df_flat = flatten_dataframe(df)
    print("Flattened dataset:")
    print(df_flat.head())

    model, tokenizer = load_model_and_tokenizer(model_name)

    sentences = df_flat["sentence"].tolist()
    target_words = df_flat["word"].tolist()
    semantic_labels = df_flat["semantic_group_id"].tolist()

    activations = get_target_activations(model, tokenizer, sentences, target_words, batch_size=4)

    for layer_idx, act_tensor in activations.items():
        print(f"Layer {layer_idx} activation shape: {act_tensor.shape}")
    
    print(semantic_labels)
    gdv_all = {}
    sorted_layers = sorted(activations.keys())
    # Process only the first num_layers layers.
    for layer_idx in sorted_layers:
        act_tensor = activations[layer_idx]  # shape: [N, hidden_dim]
        X = act_tensor.numpy()  # convert to numpy array
        gdv_value = compute_gdv(X, semantic_labels)
        gdv_all[layer_idx] = gdv_value
        print(f"Layer {layer_idx}: GDV = {gdv_value:.4f}")
        plot_layer_activations(X, semantic_labels, layer_idx, gdv_value)
    
    output_dir = 'results/gdv/'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "gdv_values.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Layer", "GDV"])
        for layer_idx, gdv in gdv_all.items():
            writer.writerow([layer_idx, gdv])
    print(f"GDV values saved to {csv_path}")
    ani = animate_layers_smooth(activations, semantic_labels, hold_count=3, interp_count=5, interval=500)

    return gdv_all



def plot_layer_activations(activations: np.ndarray, labels: np.ndarray, layer_idx: int, gdv_value: float, output_dir='results/gdv/plots'):
    """
    Uses PCA to reduce a [N, hidden_dim] array of activations into 2D and
    produces a scatter plot. Points are colored by their semantic group id.
    The computed GDV is displayed in the title.
    
    Parameters:
        activations (np.ndarray): Array of shape [N, hidden_dim].
        labels (np.ndarray): Array of semantic group labels (shape [N,]).
        layer_idx (int): The index of the layer for labeling.
        gdv_value (float): The computed GDV for this layer.
        output_dir (str): Directory in which to save the plot.
    """
    # Dimensionality reduction.
    pca = PCA(n_components=2)
    activations_2d = pca.fit_transform(activations)
    
    plt.figure(figsize=(6, 6))
    
    unique_groups = np.unique(labels)
    for group in unique_groups:
        group_mask = labels == group   # Here, labels is an array of shape [N,] matching X
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

def compute_mean_intra_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the mean Euclidean distance within each class and average these over all classes.

    Args:
        X (np.ndarray): Array of shape [N, D] containing activation vectors.
        labels (np.ndarray): Array of shape [N,] of class labels.

    Returns:
        float: Average intra-class distance.
    """
    unique_labels = np.unique(labels)
    intra_dists = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        # Skip classes with fewer than 2 samples.
        if len(idx) < 2:
            continue
        # Compute pairwise distances for the points in the class.
        dists = pdist(X[idx], metric="euclidean")
        intra_dists.append(np.mean(dists))
    if len(intra_dists) == 0:
        return 0.0
    return np.mean(intra_dists)

def compute_mean_inter_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the mean Euclidean distance between classes, averaged over all pairs of distinct classes.

    Args:
        X (np.ndarray): Array of shape [N, D] containing activation vectors.
        labels (np.ndarray): Array of shape [N,] of class labels.

    Returns:
        float: Average inter-class distance.
    """
    unique_labels = np.unique(labels)
    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            idx1 = np.where(labels == unique_labels[i])[0]
            idx2 = np.where(labels == unique_labels[j])[0]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            # Compute pairwise distances between samples in group i and group j.
            # Here we use broadcasting to compute the Euclidean distance.
            diff = X[idx1][:, None, :] - X[idx2]  # shape: [|idx1|, |idx2|, D]
            dists = np.linalg.norm(diff, axis=2)    # shape: [|idx1|, |idx2|]
            inter_dists.append(np.mean(dists))
    if len(inter_dists) == 0:
        return 0.0
    return np.mean(inter_dists)

def compute_gdv(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Generalized Discrimination Value (GDV) given a set of activation vectors and class labels.

    Args:
        X (np.ndarray): Activation vectors of shape [N, D].
        labels (np.ndarray): Semantic group labels of shape [N,].

    Returns:
        float: The computed GDV.
    """
    # Get the number of distinct classes.
    unique_labels = np.unique(labels)
    L = len(unique_labels)
    # If there is only one class, GDV is not defined.
    if L < 2:
        return 0.0
    intra = compute_mean_intra_class_distance(X, labels)
    inter = compute_mean_inter_class_distance(X, labels)
    D = X.shape[1]
    gdv = (1 / np.sqrt(D)) * ((1 / L) * intra - (2 / (L * (L - 1))) * inter)
    return gdv



def animate_layers_smooth(activations_dict: dict, semantic_labels: np.ndarray, hold_count: int = 3, interp_count: int = 5, interval: int = 500):
    """
    Creates a matplotlib animation that shows smooth transitions between recorded layers:
      - Each recorded layer is held for hold_count frames,
      - Then a smooth interpolation (with interp_count frames) is shown between layers.
    
    The title at recorded frames shows the actual layer and its GDV.
    
    Parameters:
      activations_dict: dict mapping recorded layer index -> torch.Tensor of shape [N, hidden_dim].
      semantic_labels: NumPy array (shape [N,]) containing per-sample semantic labels.
      hold_count: how many frames to hold each recorded layer.
      interp_count: how many frames to interpolate from one recorded layer to the next.
      interval: delay between frames in milliseconds.
    
    Returns:
      ani: A matplotlib.animation.FuncAnimation instance.
    """
    sorted_layers = sorted(activations_dict.keys())
    # Precompute PCA projections and GDV for each recorded layer.
    pca_data = {}
    gdv_data = {}
    for layer in sorted_layers:
        X = activations_dict[layer].numpy()
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        pca_data[layer] = X_2d
        gdv_data[layer] = compute_gdv(X, semantic_labels)
    
    # Build a frame schedule list.
    # Each frame is a tuple: (mode, base_index, fraction)
    # mode can be "recorded" or "interp"
    # For a "recorded" frame, fraction is ignored; for "interp", fraction is between 0 and 1.
    schedule = []
    L = len(sorted_layers)
    for i in range(L - 1):
        # Hold recorded layer i for hold_count frames.
        for _ in range(hold_count):
            schedule.append(("recorded", i, 0))
        # Interpolate from layer i to layer i+1.
        # We'll generate interp_count frames, with fraction going from 0 to 1.
        for j in range(1, interp_count):  # start at 1 to avoid duplicating the recorded frame.
            frac = j / (interp_count - 1)
            schedule.append(("interp", i, frac))
    # Finally, hold the final recorded layer.
    for _ in range(hold_count):
        schedule.append(("recorded", L - 1, 0))
    
    total_frames = len(schedule)
    
    # Determine unique semantic groups for coloring.
    unique_groups = np.unique(semantic_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_groups))
    
    # Set up the figure.
    fig, ax = plt.subplots(figsize=(6, 6))
    scatters = {}
    for idx, group in enumerate(unique_groups):
        scat = ax.scatter([], [], color=colors(idx), label=f"Group {group}")
        scatters[group] = scat
    title = ax.set_title("")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    
    # Fix axis limits using data from all recorded layers.
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
            # Use the recorded data for layer sorted_layers[base_idx]
            cur_layer = sorted_layers[base_idx]
            data_2d = pca_data[cur_layer]
            cur_gdv = gdv_data[cur_layer]
            title_text = f"Layer {cur_layer}: GDV = {cur_gdv:.4f}"
        elif mode == "interp":
            # Interpolate between layer sorted_layers[base_idx] and layer sorted_layers[base_idx+1]
            layer_a = sorted_layers[base_idx]
            layer_b = sorted_layers[base_idx + 1]
            data_2d = (1 - frac) * pca_data[layer_a] + frac * pca_data[layer_b]
            cur_gdv = (1 - frac) * gdv_data[layer_a] + frac * gdv_data[layer_b]
            title_text = f"Transition: {layer_a} ➔ {layer_b} (t={frac:.2f})"
        else:
            data_2d = None
            title_text = ""
        
        # Update scatter plot for each semantic group.
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