import csv
import logging
import os
import pickle
from typing import Any, Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from models import find_target_span, get_dual_position_activations, load_model_and_tokenizer
from data import flatten_dataframe


# ---- GDV Helper Functions ----

def compute_mean_intra_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    intra_dists = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            continue
        D = cdist(X[idx], X[idx], metric='euclidean')
        upper = D[np.triu_indices_from(D, k=1)]
        intra_dists.append(float(upper.mean()))
    return float(np.mean(intra_dists)) if intra_dists else 0.0


def compute_mean_inter_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            X1 = X[np.where(labels == unique_labels[i])[0]]
            X2 = X[np.where(labels == unique_labels[j])[0]]
            if len(X1) == 0 or len(X2) == 0:
                continue
            inter_dists.append(float(cdist(X1, X2, metric='euclidean').mean()))
    return float(np.mean(inter_dists)) if inter_dists else 0.0


def compute_gdv(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Geometric Discriminability Value (GDV):

        GDV = (1/√D) * ( (1/L)*intra - (2/(L*(L-1)))*inter )

    where D is the hidden dimension and L the number of classes, computed on
    z-score-normalised activations. More negative → better class separation.
    """
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    Xz = (X - mu) / sigma
    L = len(np.unique(labels))
    if L < 2:
        return 0.0
    intra = compute_mean_intra_class_distance(Xz, labels)
    inter = compute_mean_inter_class_distance(Xz, labels)
    D = Xz.shape[1]
    return float((1 / np.sqrt(D)) * ((1 / L) * intra - (2 / (L * (L - 1))) * inter))


def _key_layer_indices(sorted_layers: List[int], gdv_by_layer: Dict[int, float]) -> List[int]:
    """
    Return ~6 representative layer indices for PCA plots:
    embedding (0), quartiles (25%/50%/75%), best (most negative GDV), last.
    Avoids generating hundreds of plots for every layer.
    """
    n = len(sorted_layers)
    if n <= 7:
        return sorted_layers
    candidates = {
        sorted_layers[0],
        sorted_layers[max(1, n // 4)],
        sorted_layers[n // 2],
        sorted_layers[3 * n // 4],
        sorted_layers[-1],
        min(gdv_by_layer, key=gdv_by_layer.get),  # best layer
    }
    return sorted(candidates)


def plot_layer_activations(
    X: np.ndarray,
    labels: np.ndarray,
    layer_idx: int,
    gdv_value: float,
    output_dir: str,
) -> None:
    """PCA scatter plot of activations coloured by semantic group."""
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    for g in np.unique(labels):
        mask = labels == g
        plt.scatter(X2d[mask, 0], X2d[mask, 1], label=f"Group {g}")
    plt.title(f"Layer {layer_idx}\nGDV = {gdv_value:.4f}")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}.png"))
    plt.close()


def _compute_rank_aggregated_gdv(
    gdv_per_word: Dict[str, Dict[int, float]],
    sorted_layers: List[int],
) -> Dict[int, float]:
    """
    Rank aggregation across homonyms.

    For each word, rank layers 1..N where rank 1 = most negative GDV (best
    discrimination). Average the ranks across all words per layer. A lower
    mean rank means that layer consistently separates senses well across
    different homonyms, regardless of each word's absolute GDV magnitude.

    This is more robust than raw mean GDV when words have very different
    baseline discriminabilities (e.g. an easy vs. a hard homonym).
    """
    words = list(gdv_per_word.keys())
    # Build layer × word matrix
    matrix = np.array([[gdv_per_word[w][layer] for w in words] for layer in sorted_layers])
    # Rank each column (word) ascending: most negative = rank 1
    ranks = np.argsort(np.argsort(matrix, axis=0), axis=0) + 1  # 1-based
    mean_ranks = ranks.mean(axis=1)
    return {layer: float(mean_ranks[i]) for i, layer in enumerate(sorted_layers)}


def save_target_activations(
    base_dir: str,
    word: str,
    model_name: str,
    activations: Dict[int, torch.Tensor],
    labels: np.ndarray,
    sentences: np.ndarray,
    words: np.ndarray,
    subdir: str = 'activations',
) -> None:
    """
    Save per-layer activations to HDF5 files.

    subdir : "activations" (homonym-token position, default) or
             "activations_final" (final-content-token position — see
             get_dual_position_activations / H4's final-position centroids).
    """
    target_dir = os.path.join(base_dir, subdir, word, model_name.replace('/', '_'))
    os.makedirs(target_dir, exist_ok=True)
    for layer_idx, tensor in activations.items():
        arr = tensor.cpu().numpy()
        h5_file = os.path.join(target_dir, f"layer_{layer_idx}.h5")
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('X', data=arr)
            f.create_dataset('labels', data=labels)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('sentences',
                             data=np.array([s.encode('utf-8') for s in sentences]), dtype=dt)
            f.create_dataset('words',
                             data=np.array([w.encode('utf-8') for w in words]), dtype=dt)
            f.create_dataset('mu', data=arr.mean(axis=0))
            f.create_dataset('sigma', data=arr.std(axis=0) + 1e-12)


def run_gdv_experiment(
    df: pd.DataFrame,
    model_name: str,
    model_type: str = "default",
) -> Dict[int, float]:
    """
    Main experiment loop.

    For each word in the dataframe:
      - Extract target-word activations per sentence per layer.
      - Compute GDV per layer, save PCA plots and per-word CSV.
      - Cache activations to HDF5.

    Final `gdv_values.csv` contains the mean GDV across all words per layer.

    Pooling strategy is auto-detected:
      - Bidirectional encoders → 'target' (mean-pool homonym subword tokens).
      - Causal/decoder-only   → 'last_token' (last non-padding token captures
        full left context, which is more informative than the homonym position
        that may precede the disambiguating context).
    """
    df_flat = flatten_dataframe(df)

    # Drop sentences where the target word (or its plain plural) does not occur
    # as its own word (e.g. "lighter", "Sunlight" for target "light" — but not
    # "lights", which is kept). Profiling centroids must be built from clean
    # occurrences of the target word/sense; silently falling back to a zero
    # vector for these would pollute the centroid.
    has_span = df_flat.apply(
        lambda row: find_target_span(row["sentence"], row["word"]) is not None, axis=1
    )
    n_dropped = int((~has_span).sum())
    if n_dropped:
        dropped_words = sorted(df_flat.loc[~has_span, "word"].unique())
        logging.warning(
            "Dropping %d/%d profiling sentences with no word-boundary match for words: %s",
            n_dropped, len(df_flat), dropped_words,
        )
    df_flat = df_flat[has_span].reset_index(drop=True)

    sentences    = np.array(df_flat['sentence'].tolist(), dtype=object)
    target_words = np.array(df_flat['word'].tolist(), dtype=object)
    labels       = np.array(df_flat['semantic_group_id'].tolist(), dtype=int)

    model, tokenizer = load_model_and_tokenizer(model_name, model_type=model_type)

    # Always use target-token pooling for profiling (and everything derived from
    # `activations` below — GDV, plots, H1/H2/H3 centroids) so that centroids are
    # built from the homonym token's representation, consistent with all
    # downstream tests. Causal models will only see left context at the homonym
    # position — that is intentional and correctly reflects the decoder's
    # architectural limitation.
    #
    # get_dual_position_activations gets the final-content-token representation
    # (skipping trailing special tokens) in the same forward pass at no extra
    # compute cost. Those are cached separately below (activations_final/) to
    # build final-position centroids for H4's encoder redesign — H4 must score
    # final-token activations against centroids from the same position, not
    # against homonym-position centroids from a different subspace.
    activations, final_activations = get_dual_position_activations(
        model, tokenizer,
        sentences.tolist(),
        target_words.tolist(),
        batch_size=4,
    )

    safe_model_name = model_name.replace('/', '_')
    output_dir = f"results/{safe_model_name}_gdv/"
    os.makedirs(output_dir, exist_ok=True)
    sorted_layers = sorted(activations.keys())
    unique_words  = sorted(set(target_words.tolist()))

    gdv_per_word: Dict[str, Dict[int, float]] = {}

    for word in unique_words:
        mask           = target_words == word
        word_sentences = sentences[mask]
        word_labels    = labels[mask]
        word_acts      = {layer: activations[layer][mask] for layer in sorted_layers}

        save_target_activations(
            base_dir='results',
            word=word,
            model_name=model_name,
            activations=word_acts,
            labels=word_labels,
            sentences=word_sentences,
            words=target_words[mask],
        )

        word_final_acts = {layer: final_activations[layer][mask] for layer in sorted_layers}
        save_target_activations(
            base_dir='results',
            word=word,
            model_name=model_name,
            activations=word_final_acts,
            labels=word_labels,
            sentences=word_sentences,
            words=target_words[mask],
            subdir='activations_final',
        )

        word_gdv: Dict[int, float] = {}
        for layer_idx in sorted_layers:
            X = word_acts[layer_idx].cpu().numpy()
            word_gdv[layer_idx] = compute_gdv(X, word_labels)
        gdv_per_word[word] = word_gdv

        # Only plot key layers (~6 per word) to avoid generating hundreds of files
        key_layers = _key_layer_indices(sorted_layers, word_gdv)
        for layer_idx in key_layers:
            X = word_acts[layer_idx].cpu().numpy()
            plot_layer_activations(
                X, word_labels, layer_idx, word_gdv[layer_idx],
                os.path.join(output_dir, 'plots', word),
            )

        word_csv = os.path.join(output_dir, f"gdv_values_{word}.csv")
        with open(word_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Layer', 'GDV'])
            for layer_idx, val in word_gdv.items():
                writer.writerow([layer_idx, val])
        logging.info("Per-word GDV CSV saved to %s", word_csv)

    # Aggregate: mean GDV across words
    gdv_all: Dict[int, float] = {
        layer: float(np.mean([gdv_per_word[w][layer] for w in unique_words]))
        for layer in sorted_layers
    }

    csv_path = os.path.join(output_dir, 'gdv_values.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'GDV'])
        for layer_idx, val in gdv_all.items():
            writer.writerow([layer_idx, val])
    logging.info("Aggregate GDV CSV saved to %s", csv_path)

    # Rank aggregation: more robust summary when homonyms differ in base difficulty
    gdv_rank = _compute_rank_aggregated_gdv(gdv_per_word, sorted_layers)
    rank_csv_path = os.path.join(output_dir, 'gdv_rank_aggregated.csv')
    with open(rank_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'MeanRank'])
        for layer_idx, val in gdv_rank.items():
            writer.writerow([layer_idx, val])
    logging.info("Rank-aggregated GDV CSV saved to %s", rank_csv_path)

    # Build PCA coords per word for Dash visualisation
    pca_data: Dict[str, Dict[int, np.ndarray]] = {}
    for word in unique_words:
        pca_data[word] = {}
        mask = target_words == word
        for layer_idx in sorted_layers:
            X = activations[layer_idx][mask].cpu().numpy()
            pca_data[word][layer_idx] = PCA(n_components=2).fit_transform(X)

    layer_info: Dict[int, Any] = {
        layer: {
            word: {
                'x':        pca_data[word][layer][:, 0],
                'y':        pca_data[word][layer][:, 1],
                'group':    labels[target_words == word],
                'sentence': sentences[target_words == word],
            }
            for word in unique_words
        }
        for layer in sorted_layers
    }

    meta = {
        'model_name':    model_name,
        'words':         unique_words,
        'pooling':       "target",
        'total_layers':  len(sorted_layers),
        'gdv_per_layer': gdv_all,
        'gdv_per_word':  gdv_per_word,
        'gdv_rank':      gdv_rank,
    }
    output_data = {
        'sorted_layers': sorted_layers,
        'layer_data':    layer_info,
        'gdv_per_layer': gdv_all,
        'gdv_per_word':  gdv_per_word,
        'gdv_rank':      gdv_rank,
        'meta':          meta,
    }
    pkl_path = os.path.join(output_dir, f"{safe_model_name}_gdv.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(output_data, f)
    logging.info("Dash data saved to %s", pkl_path)

    return gdv_all
