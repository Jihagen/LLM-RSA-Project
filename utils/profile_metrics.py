import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


def compute_mean_intra_class_distance(
    X: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    unique_labels = np.unique(labels)
    intra_dists: List[float] = []

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            continue

        dists: List[float] = []
        wts: List[float] = []
        for i, pos_i in enumerate(idx[:-1]):
            for pos_j in idx[i + 1 :]:
                dist = float(np.linalg.norm(X[pos_i] - X[pos_j]))
                weight = float(weights[pos_i] * weights[pos_j]) if weights is not None else 1.0
                dists.append(dist * weight)
                wts.append(weight)

        if wts and sum(wts) > 0:
            intra_dists.append(sum(dists) / sum(wts))

    return float(np.mean(intra_dists)) if intra_dists else 0.0


def compute_mean_inter_class_distance(
    X: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    unique_labels = np.unique(labels)
    inter_dists: List[float] = []

    for idx_a, label_a in enumerate(unique_labels[:-1]):
        label_b_candidates = unique_labels[idx_a + 1 :]
        samples_a = np.where(labels == label_a)[0]

        for label_b in label_b_candidates:
            samples_b = np.where(labels == label_b)[0]
            if len(samples_a) == 0 or len(samples_b) == 0:
                continue

            dists: List[float] = []
            wts: List[float] = []
            for pos_a in samples_a:
                for pos_b in samples_b:
                    dist = float(np.linalg.norm(X[pos_a] - X[pos_b]))
                    weight = float(weights[pos_a] * weights[pos_b]) if weights is not None else 1.0
                    dists.append(dist * weight)
                    wts.append(weight)

            if wts and sum(wts) > 0:
                inter_dists.append(sum(dists) / sum(wts))

    return float(np.mean(inter_dists)) if inter_dists else 0.0


def compute_gdv(
    X: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    if X.ndim != 2:
        raise ValueError(f"GDV expects a 2D matrix, got shape {X.shape}")

    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    Xz = ((X - mu) / sigma) * 0.5

    if weights is None:
        weights = np.ones(Xz.shape[0], dtype=float)

    n_classes = len(np.unique(labels))
    if n_classes < 2:
        return 0.0

    intra = compute_mean_intra_class_distance(Xz, labels, weights)
    inter = compute_mean_inter_class_distance(Xz, labels, weights)
    dim = Xz.shape[1]

    return float((1.0 / np.sqrt(dim)) * ((1.0 / n_classes) * intra - (2.0 / (n_classes * (n_classes - 1))) * inter))


def compute_profile_sharpness(scores: Sequence[float], peak_index: int) -> float:
    if not scores:
        return 0.0
    if peak_index < 0 or peak_index >= len(scores):
        raise IndexError("peak_index is out of range for the provided score sequence.")

    peak_value = float(scores[peak_index])
    if len(scores) == 1:
        return peak_value

    remainder = np.array([value for idx, value in enumerate(scores) if idx != peak_index], dtype=float)
    return float((peak_value - remainder.mean()) / (remainder.std() + 1e-12))


def top_k_layers(
    layers: Sequence[int],
    scores: Sequence[float],
    k: int = 3,
    maximize: bool = True,
) -> List[int]:
    if len(layers) != len(scores):
        raise ValueError("layers and scores must have the same length.")

    scored_layers = list(zip(layers, scores))
    scored_layers.sort(key=lambda item: item[1], reverse=maximize)
    return [int(layer) for layer, _ in scored_layers[:k]]


def compute_semantic_band(peaks: Sequence[int], band_width: int = 3) -> Dict[str, Union[float, int, List[int], None]]:
    if not peaks:
        return {
            "band_start": None,
            "band_end": None,
            "band_layers": [],
            "support_count": 0,
            "support_fraction": 0.0,
        }

    unique_peaks = sorted(set(int(peak) for peak in peaks))
    if band_width <= 0:
        raise ValueError("band_width must be positive.")

    peak_array = np.array(peaks, dtype=int)
    best_band = None
    best_support = -1

    for start in range(min(unique_peaks), max(unique_peaks) + 1):
        band_layers = list(range(start, start + band_width))
        support = int(np.isin(peak_array, band_layers).sum())
        if support > best_support:
            best_support = support
            best_band = band_layers

    if best_band is None:
        best_band = [unique_peaks[0]]
        best_support = int((peak_array == unique_peaks[0]).sum())

    return {
        "band_start": int(best_band[0]),
        "band_end": int(best_band[-1]),
        "band_layers": [int(layer) for layer in best_band],
        "support_count": int(best_support),
        "support_fraction": float(best_support / len(peaks)),
    }


def maybe_global_layer(peaks: Sequence[int], min_fraction: float = 0.6) -> Optional[int]:
    if not peaks:
        return None

    counts: Dict[int, int] = {}
    for peak in peaks:
        counts[int(peak)] = counts.get(int(peak), 0) + 1

    global_layer, count = max(counts.items(), key=lambda item: item[1])
    if count / len(peaks) >= min_fraction:
        return int(global_layer)
    return None


def interpolate_profile(
    layers: Sequence[int],
    scores: Sequence[float],
    num_points: int = 64,
) -> np.ndarray:
    if len(layers) != len(scores):
        raise ValueError("layers and scores must have the same length.")
    if len(layers) == 1:
        return np.repeat(float(scores[0]), num_points)

    layers_arr = np.array(layers, dtype=float)
    scores_arr = np.array(scores, dtype=float)

    normalized_layers = layers_arr / max(layers_arr.max(), 1.0)
    target_grid = np.linspace(0.0, 1.0, num_points)
    return np.interp(target_grid, normalized_layers, scores_arr)


def relative_peak_position(layers: Sequence[int], peak_layer: int) -> float:
    if not layers:
        return 0.0
    max_layer = max(int(layer) for layer in layers)
    if max_layer == 0:
        return 0.0
    return float(peak_layer / max_layer)


def safe_pearsonr(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    arr_a = np.array(values_a, dtype=float)
    arr_b = np.array(values_b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("Both sequences must have the same shape.")
    if arr_a.size == 0:
        return 0.0
    if np.allclose(arr_a, arr_a[0]) or np.allclose(arr_b, arr_b[0]):
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("linear_cka expects 2D matrices.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    cross_cov = Y_centered.T @ X_centered
    hsic = float(np.linalg.norm(cross_cov, ord="fro") ** 2)
    norm_x = float(np.linalg.norm(X_centered.T @ X_centered, ord="fro"))
    norm_y = float(np.linalg.norm(Y_centered.T @ Y_centered, ord="fro"))

    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return hsic / (norm_x * norm_y)
