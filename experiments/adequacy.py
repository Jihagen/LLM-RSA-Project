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


# ── Word-alone baseline (used in H0) ─────────────────────────────────────────

def compute_word_alone_margins(
    model,
    tokenizer,
    words: List[str],
    centroids: Dict[int, Dict[int, np.ndarray]],
    layer_idx: int,
    correct_sense: int = 0,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Feed each word as a standalone sentence ("bank", "bat", …) through the model
    and compute M_l at the word token position.

    This establishes the model's prior before any sentence context is added.
    In theory it should be near zero because a bare token has no disambiguating
    context — any deviation reveals an intrinsic lexical bias in the embedding.

    Returns {word: M_l_word_alone}.
    """
    from models import get_target_activations

    wrong_sense = 1 - correct_sense
    if layer_idx not in centroids:
        return {}
    c_correct = centroids[layer_idx][correct_sense]
    c_wrong   = centroids[layer_idx][wrong_sense]

    # Build minimal single-word sentences; one per unique word
    unique_words = list(dict.fromkeys(words))
    sentences    = unique_words  # the word is its own sentence

    acts = get_target_activations(
        model, tokenizer,
        sentences, unique_words,
        batch_size=batch_size,
        layer_indices=[layer_idx],
        pooling="target",
    )
    H = acts[layer_idx].numpy()

    return {
        word: float(adequacy_margin(H[i], c_correct, c_wrong))
        for i, word in enumerate(unique_words)
    }


# ── GDV-best layer identification (used in H2) ────────────────────────────────

def gdv_best_layer(gdv_csv_path: str) -> int:
    """
    Return the layer index with the most negative GDV from a gdv_values.csv.

    Layer 0 (pre-transformer embeddings) is excluded — see adequacy_best_layer
    for why a non-contextual layer cannot be a valid "best" answer here either.
    """
    layers, values = [], []
    with open(gdv_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["Layer"])
            if layer == 0:
                continue
            layers.append(layer)
            values.append(float(row["GDV"]))
    return layers[int(np.argmin(values))]


def adequacy_best_layer(profile: Dict[int, Dict]) -> int:
    """
    Return the layer with the highest mean adequacy margin.

    Layer 0 (the pre-transformer embedding layer) is excluded from the search:
    it carries no contextual mixing, so any separation there reflects token-identity
    or positional artifacts rather than sense disambiguation, and is not a valid
    "best contextual layer" answer even if its in-sample margin happens to be highest.
    """
    candidates = [l for l in profile if l != 0] or list(profile)
    return max(candidates, key=lambda l: profile[l]["mean"])


# ── I/O helpers ───────────────────────────────────────────────────────────────

def compute_carrier_margins(
    model,
    tokenizer,
    carrier_items: List[Dict],
    centroids: Dict[int, Dict[int, np.ndarray]],
    layer_idx: int,
    batch_size: int = 8,
) -> List[Dict]:
    """
    Run carrier sentences alone through the model and compute M_l at the
    homonym-token position. Returns one record per (carrier, sense) pair.

    carrier_items : list of {'word': str, 'carrier': str, 'sense': int}

    The carrier M_l is the model's prior for that template sentence.
    Use it to compute context_gain = M_l(full_sentence) - M_l(carrier_alone).
    """
    import torch
    from models import get_target_activations, is_decoder_only

    # Carriers are short sentences — always use target pooling regardless of arch
    sentences = [item["carrier"] for item in carrier_items]
    words     = [item["word"]    for item in carrier_items]

    acts = get_target_activations(
        model, tokenizer,
        sentences, words,
        batch_size=batch_size,
        layer_indices=[layer_idx],
        pooling="target",
    )
    H = acts[layer_idx].numpy()

    records = []
    for i, item in enumerate(carrier_items):
        sense   = item["sense"]
        c_sense = sense
        w_sense = 1 - sense  # assumes binary 0/1 senses
        if c_sense not in centroids[layer_idx] or w_sense not in centroids[layer_idx]:
            continue
        m = adequacy_margin(H[i], centroids[layer_idx][c_sense], centroids[layer_idx][w_sense])
        records.append({
            "word":         item["word"],
            "carrier":      item["carrier"],
            "sense":        sense,
            "M_l_carrier":  round(m, 4),
            "biased":       abs(m) > 0.3,  # flag strong priors
            "layer":        layer_idx,
        })
    return records


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
