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
    """
    Vectorised adequacy margins for a matrix H [N, D] against a SINGLE fixed
    (c_correct, c_wrong) pair. Only valid when every row in H shares the same
    intended sense. If a batch mixes rows of both senses, use
    symmetric_adequacy_margins instead — applying one fixed centroid pair to a
    mixed-sense batch gives the wrong sign for whichever rows are not the
    assumed "correct" sense, and any mean/fraction computed over that mixture
    is not a meaningful adequacy statistic (it converges toward ~0 / ~50% by
    construction even when the representation separates the senses perfectly).
    """
    return np.linalg.norm(H - c_wrong, axis=1) - np.linalg.norm(H - c_correct, axis=1)


def symmetric_adequacy_margins(
    X: np.ndarray,
    labels: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
) -> np.ndarray:
    """
    Per-row adequacy margin using each row's OWN true sense label to decide
    which centroid is "correct" and which is "wrong" for that row.

    Row with label 0: margin = d(row, c1) - d(row, c0)  (positive = correct)
    Row with label 1: margin = d(row, c1) - d(row, c0), negated -> d(row,c0)-d(row,c1)

    Use this whenever a batch contains both senses (H1 profiling sets, H3/H4
    paired-sentence stimuli, which are balanced 50/50 across senses per word).
    A representation that separates the two senses perfectly gives ALL rows a
    positive margin under this definition, unlike batch_adequacy_margins applied
    naively to a mixed batch.
    """
    labels = np.asarray(labels)
    d0 = np.linalg.norm(X - c0, axis=1)
    d1 = np.linalg.norm(X - c1, axis=1)
    return np.where(labels == 0, d1 - d0, d0 - d1)


# ── Normalized (cross-architecture-comparable) margins ─────────────────────────
#
# Raw M_l is not comparable across architectures or layers: decoder-only LLMs
# develop a handful of very-high-magnitude "massive activation" hidden
# dimensions concentrated in later layers, which inflate ||h - c|| distances
# (and thus raw M_l) by 1-2 orders of magnitude for reasons unrelated to sense
# adequacy. Sign and fraction-adequate are robust to this (they only use the
# sign of M_l), but raw means are not. Dividing by the inter-centroid distance
# ||c_correct - c_wrong|| at that layer rescales M_l into a comparable range:
# +1 means h sits exactly at c_correct, -1 means it sits exactly at c_wrong,
# 0 is the decision boundary — and the ratio cancels out any shared magnitude
# inflation common to h, c_correct, and c_wrong.

def normalized_adequacy_margin(
    h: np.ndarray,
    c_correct: np.ndarray,
    c_wrong: np.ndarray,
) -> float:
    """Adequacy margin for a single hidden-state vector, scaled by inter-centroid distance."""
    scale = float(np.linalg.norm(c_correct - c_wrong)) + 1e-12
    return adequacy_margin(h, c_correct, c_wrong) / scale


def symmetric_normalized_adequacy_margins(
    X: np.ndarray,
    labels: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
) -> np.ndarray:
    """Normalized counterpart of symmetric_adequacy_margins — see module note above."""
    scale = float(np.linalg.norm(c0 - c1)) + 1e-12
    return symmetric_adequacy_margins(X, labels, c0, c1) / scale


# ── Centroid loading from H5 activation cache ─────────────────────────────────

def load_centroids(
    results_dir: str,
    model_name: str,
    word: str,
    subdir: str = "activations",
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Load per-layer sense centroids from cached H5 activation files.

    subdir : "activations" (homonym-token position, default) or
             "activations_final" (final-content-token position — see
             load_final_centroids).

    Returns
    -------
    {layer_idx: {sense_id: centroid_array [D]}}
    """
    safe_model = model_name.replace("/", "_")
    h5_dir = Path(results_dir) / subdir / word / safe_model

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


def load_final_centroids(
    results_dir: str,
    model_name: str,
    word: str,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Load per-layer sense centroids computed at the final-content-token
    position (activations_final/), rather than the homonym-token position.

    H4 must score final-token activations against these, not against
    load_centroids()'s homonym-position centroids: those live in a different
    representational subspace for encoders, which is why scoring final-token
    activations against them collapsed frac_adeq_final to exactly chance
    (0.500 in 16/16 model/word cells) regardless of true separability.
    """
    return load_centroids(results_dir, model_name, word, subdir="activations_final")


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
    Each sentence's margin is computed against ITS OWN true sense as "correct"
    (via symmetric_adequacy_margins) — profiling sets contain both senses for
    a word, so a single fixed correct/wrong centroid pair would silently give
    the wrong sign for half the sentences.

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

        c0 = X[labels == 0].mean(axis=0)
        c1 = X[labels == 1].mean(axis=0)

        margins      = symmetric_adequacy_margins(X, labels, c0, c1)
        margins_norm = symmetric_normalized_adequacy_margins(X, labels, c0, c1)

        # Per-sense margins (diagnostic only — both should now be positive
        # when the representation separates the senses well, since each
        # row's margin is already relative to its own true sense)
        margins_sense0 = margins[labels == 0]
        margins_sense1 = margins[labels == 1]

        profile[layer_idx] = {
            "margins":            margins,
            "mean":               float(margins.mean()),
            "mean_norm":          float(margins_norm.mean()),
            "fraction_adequate":  float((margins > epsilon).mean()),
            "mean_margin_sense0": float(margins_sense0.mean()),
            "mean_margin_sense1": float(margins_sense1.mean()),
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
        writer.writerow(["Layer", "MeanMargin", "MeanMarginNorm", "FractionAdequate",
                         "MeanMarginSense0", "MeanMarginSense1"])
        for layer_idx in sorted(profile):
            r = profile[layer_idx]
            writer.writerow([
                layer_idx,
                round(r["mean"], 6),
                round(r.get("mean_norm", float("nan")), 6),
                round(r["fraction_adequate"], 4),
                round(r.get("mean_margin_sense0", float("nan")), 6),
                round(r.get("mean_margin_sense1", float("nan")), 6),
            ])
    logger.info("Saved adequacy profile to %s", output_path)
