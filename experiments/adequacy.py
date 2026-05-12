"""
Core functions for the adequacy margin measure M_l.

    M_l = d(h_l, c_wrong) - d(h_l, c_correct)

Positive  → representation is closer to the correct sense centroid (adequate).
Zero      → on the decision boundary.
Negative  → closer to the wrong sense centroid (inadequate).

Adequacy threshold: M_l > epsilon  (default epsilon = 0).
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# ── Instance-level measures ────────────────────────────────────────────────────

def adequacy_margin(
    h: np.ndarray,
    c_correct: np.ndarray,
    c_wrong: np.ndarray,
) -> float:
    """Adequacy margin for a single hidden-state vector h [D]."""
    return float(np.linalg.norm(h - c_wrong) - np.linalg.norm(h - c_correct))


def batch_adequacy_margins(
    H: np.ndarray,
    c_correct: np.ndarray,
    c_wrong: np.ndarray,
) -> np.ndarray:
    """Vectorised adequacy margins for a matrix H [N, D]. Returns array [N]."""
    return np.linalg.norm(H - c_wrong, axis=1) - np.linalg.norm(H - c_correct, axis=1)


# ── Centroid loading from H5 activation cache ─────────────────────────────────

def load_centroids(
    results_dir: str,
    model_name: str,
    word: str,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Load per-layer sense centroids from cached H5 activation files.

    Returns
    -------
    {layer_idx: {sense_id: centroid_array [D]}}
    """
    safe_model = model_name.replace("/", "_")
    h5_dir = Path(results_dir) / "activations" / word / safe_model

    if not h5_dir.exists():
        raise FileNotFoundError(
            f"No activation cache for model={model_name!r}, word={word!r} at {h5_dir}. "
            "Run the GDV profiling step first."
        )

    centroids: Dict[int, Dict[int, np.ndarray]] = {}
    h5_files = sorted(
        h5_dir.glob("layer_*.h5"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not h5_files:
        raise FileNotFoundError(f"No layer H5 files found in {h5_dir}.")

    for h5_path in h5_files:
        layer_idx = int(h5_path.stem.split("_")[1])
        with h5py.File(h5_path, "r") as f:
            X      = f["X"][:]
            labels = f["labels"][:]
        centroids[layer_idx] = {
            int(s): X[labels == int(s)].mean(axis=0)
            for s in np.unique(labels)
        }

    logger.debug("Loaded %d-layer centroids for %s / %s", len(centroids), model_name, word)
    return centroids


def load_all_word_centroids(
    results_dir: str,
    model_name: str,
    words: List[str],
) -> Dict[str, Dict[int, Dict[int, np.ndarray]]]:
    """Load centroids for multiple words. Returns {word: centroids}."""
    return {w: load_centroids(results_dir, model_name, w) for w in words}


# ── Layer adequacy profile (H1) ───────────────────────────────────────────────

def layer_adequacy_profile(
    results_dir: str,
    model_name: str,
    word: str,
    correct_sense: int = 0,
    epsilon: float = 0.0,
) -> Dict[int, Dict]:
    """
    Compute adequacy margin statistics per layer from cached activations.

    Centroids are computed from the full profiling set; margins are evaluated
    on the same sentences (in-sample — appropriate for H1 layer profiling).

    Returns
    -------
    {layer_idx: {'margins': np.ndarray, 'mean': float, 'fraction_adequate': float}}
    """
    wrong_sense = 1 - correct_sense

    safe_model = model_name.replace("/", "_")
    h5_dir     = Path(results_dir) / "activations" / word / safe_model
    h5_files   = sorted(h5_dir.glob("layer_*.h5"), key=lambda p: int(p.stem.split("_")[1]))

    profile: Dict[int, Dict] = {}
    for h5_path in h5_files:
        layer_idx = int(h5_path.stem.split("_")[1])
        with h5py.File(h5_path, "r") as f:
            X      = f["X"][:]
            labels = f["labels"][:]

        unique = np.unique(labels).astype(int)
        if correct_sense not in unique or wrong_sense not in unique:
            continue

        c_correct = X[labels == correct_sense].mean(axis=0)
        c_wrong   = X[labels == wrong_sense].mean(axis=0)

        margins = batch_adequacy_margins(X, c_correct, c_wrong)

        # Per-sense margins
        margins_correct = margins[labels == correct_sense]
        margins_wrong   = margins[labels == wrong_sense]

        profile[layer_idx] = {
            "margins":            margins,
            "mean":               float(margins.mean()),
            "fraction_adequate":  float((margins > epsilon).mean()),
            # Sense-stratified means (sense-0 should be positive, sense-1 negative for correct labelling)
            "mean_correct_sense": float(margins_correct.mean()),
            "mean_wrong_sense":   float(margins_wrong.mean()),
        }

    return profile


# ── GDV-best layer identification (used in H2) ────────────────────────────────

def gdv_best_layer(gdv_csv_path: str) -> int:
    """Return the layer index with the most negative GDV from a gdv_values.csv."""
    layers, values = [], []
    with open(gdv_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            layers.append(int(row["Layer"]))
            values.append(float(row["GDV"]))
    return layers[int(np.argmin(values))]


def adequacy_best_layer(profile: Dict[int, Dict]) -> int:
    """Return the layer with the highest mean adequacy margin."""
    return max(profile, key=lambda l: profile[l]["mean"])


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_profile_csv(profile: Dict[int, Dict], output_path: str) -> None:
    """Write per-layer adequacy statistics to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "MeanMargin", "FractionAdequate",
                         "MeanCorrectSense", "MeanWrongSense"])
        for layer_idx in sorted(profile):
            r = profile[layer_idx]
            writer.writerow([
                layer_idx,
                round(r["mean"], 6),
                round(r["fraction_adequate"], 4),
                round(r.get("mean_correct_sense", float("nan")), 6),
                round(r.get("mean_wrong_sense",   float("nan")), 6),
            ])
    logger.info("Saved adequacy profile to %s", output_path)
