import os
import pickle
from typing import Dict, Any
import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import csv
import pandas as pd

from dotenv import load_dotenv
from models import load_model_and_tokenizer, get_target_activations
from data import flatten_dataframe

# ---- GDV Helper Functions ----

def compute_mean_intra_class_distance(
    X: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray = None
) -> float:
    unique_labels = np.unique(labels)
    intra_dists = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            continue
        pairs = [(i, j) for i in idx for j in idx if i < j]
        dists, wts = [], []
        for i, j in pairs:
            d = np.linalg.norm(X[i] - X[j])
            w = (weights[i] * weights[j]) if weights is not None else 1.0
            dists.append(d * w)
            wts.append(w)
        if sum(wts) > 0:
            intra_dists.append(sum(dists) / sum(wts))
    return float(np.mean(intra_dists)) if intra_dists else 0.0


def compute_mean_inter_class_distance(
    X: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray = None
) -> float:
    unique_labels = np.unique(labels)
    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            idx1 = np.where(labels == unique_labels[i])[0]
            idx2 = np.where(labels == unique_labels[j])[0]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            pairs = [(ii, jj) for ii in idx1 for jj in idx2]
            dists, wts = [], []
            for ii, jj in pairs:
                d = np.linalg.norm(X[ii] - X[jj])
                w = (weights[ii] * weights[jj]) if weights is not None else 1.0
                dists.append(d * w)
                wts.append(w)
            if sum(wts) > 0:
                inter_dists.append(sum(dists) / sum(wts))
    return float(np.mean(inter_dists)) if inter_dists else 0.0


def compute_gdv(
    X: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray = None
) -> float:
    """
    Compute the GDV for raw activations X with labels, optionally weighted.
    """
    # Step 1: z-score each dim
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    Xz = (X - mu) / sigma
    # Step 2: scale by 0.5
    Xz *= 0.5
    # Default weights to 1
    if weights is None:
        weights = np.ones(Xz.shape[0], dtype=float)
    # Compute intra/inter distances
    L = len(np.unique(labels))
    if L < 2:
        return 0.0
    intra = compute_mean_intra_class_distance(Xz, labels, weights)
    inter = compute_mean_inter_class_distance(Xz, labels, weights)
    D = Xz.shape[1]
    # Step 3: combine
    gdv = (1 / np.sqrt(D)) * ((1 / L) * intra - (2 / (L * (L - 1))) * inter)
    return float(gdv)


def plot_layer_activations(
    X: np.ndarray,
    labels: np.ndarray,
    layer_idx: int,
    gdv_value: float,
    output_dir: str
) -> None:
    """
    PCA-plot and save layer activations colored by label.
    """
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    for g in np.unique(labels):
        mask = labels == g
        plt.scatter(X2d[mask, 0], X2d[mask, 1], label=f"Group {g}")
    plt.title(f"Layer {layer_idx}\nGDV = {gdv_value:.4f}")
    plt.xlabel("PC 1"); plt.ylabel("PC 2")
    plt.legend(); plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}.png"))
    plt.close()
    print(f"Plot saved to {output_dir}/layer_{layer_idx}.png")


def save_target_activations(
    base_dir: str,
    word: str,
    model_name: str,
    activations: Dict[int, torch.Tensor],
    labels: np.ndarray,
    sentences: np.ndarray,
    words: np.ndarray
) -> None:
    """
    Save per-layer activations only for the target word tokens into
    results/activations/{word}/{model_name}/layer_{i}.h5
    """
    target_dir = os.path.join(base_dir, 'activations', word, model_name.replace('/', '_'))
    os.makedirs(target_dir, exist_ok=True)
    for layer_idx, tensor in activations.items():
        arr = tensor.cpu().numpy()  # shape (N_tokens, D)
        h5_file = os.path.join(target_dir, f"layer_{layer_idx}.h5")
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('X', data=arr)
            f.create_dataset('labels', data=labels)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('sentences', data=sentences.astype('S'), dtype=dt)
            f.create_dataset('words', data=words.astype('S'), dtype=dt)
            mu = arr.mean(axis=0)
            sigma = arr.std(axis=0) + 1e-12
            f.create_dataset('mu', data=mu)
            f.create_dataset('sigma', data=sigma)
        print(f"Saved activations to {h5_file}")


def run_gdv_experiment(df: pd.DataFrame, model_name: str) -> Dict[int, float]:
    """
    Main experiment loop: flatten data, get target activations,
    save homonym-token activations, compute GDV & PCA plots,
    and save CSV + pickle summaries.
    """
    # Flatten dataset\    
    df_flat = flatten_dataframe(df)
    sentences = np.array(df_flat['sentence'].tolist(), dtype=object)
    target_words = np.array(df_flat['word'].tolist(), dtype=object)
    labels = np.array(df_flat['semantic_group_id'].tolist(), dtype=int)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Extract only the target-word activations
    activations = get_target_activations(
        model, tokenizer,
        sentences.tolist(),
        target_words.tolist(),
        batch_size=4
    )  # dict: layer_idx -> [N_tokens, D]

    # Save these token-level activations per layer
    save_target_activations(
        base_dir='results',
        word=target_words[0],
        model_name=model_name,
        activations=activations,
        labels=labels,
        sentences=sentences,
        words=target_words
    )

    # Prepare outputs
    output_dir = f"results/{model_name}_gdv/"
    os.makedirs(output_dir, exist_ok=True)

    # Compute GDV and PCA plots
    gdv_all: Dict[int, float] = {}
    pca_data: Dict[int, np.ndarray] = {}
    sorted_layers = sorted(activations.keys())
    for layer_idx in sorted_layers:
        X = activations[layer_idx].cpu().numpy()
        gdv_val = compute_gdv(X, labels)
        gdv_all[layer_idx] = gdv_val
        pca = PCA(n_components=2)
        X2d = pca.fit_transform(X)
        pca_data[layer_idx] = X2d
        plot_layer_activations(X, labels, layer_idx, gdv_val, output_dir + 'plots/')

    # Save GDV CSV
    csv_path = os.path.join(output_dir, 'gdv_values.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'GDV'])
        for layer_idx, val in gdv_all.items():
            writer.writerow([layer_idx, val])
    print(f"GDV CSV saved to {csv_path}")

    # Save Dash-friendly pickle
    meta = {
        'model_name': model_name,
        'word': target_words[0],
        'total_layers': len(sorted_layers),
        'gdv_per_layer': gdv_all
    }
    layer_info: Dict[int, Any] = {}
    for layer in sorted_layers:
        coords = pca_data[layer]
        layer_info[layer] = {
            'x': coords[:, 0],
            'y': coords[:, 1],
            'group': labels,
            'sentence': sentences
        }
    output_data = {
        'sorted_layers': sorted_layers,
        'layer_data': layer_info,
        'gdv_per_layer': gdv_all,
        'meta': meta
    }
    pkl_path = os.path.join(output_dir, f"{model_name}_{target_words[0]}_gdv.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"Dash data saved to {pkl_path}")

    return gdv_all
