"""
H1 — Layer Adequacy
====================
Hypothesis: Word-sense adequacy (M_l) differs across layers; the last layer is
not always the most adequate.

Method
------
For each (model, word) with cached activations:
  1. Compute the layer profile with leave-one-sentence-out centroids.
  2. Estimate best-versus-last performance with nested leave-one-out:
     select the best layer on each outer fold's training sentences, then score
     the untouched sentence at that layer and at the final layer.
  3. Retain a full-data leave-one-out best layer only for selecting a layer in
     genuinely separate downstream stimuli (H3-H5), not as H1 performance.

Output
------
results/study/H1/{safe_model}/h1_{word}.csv   — per-layer adequacy profile
results/study/H1/{safe_model}/h1_summary.csv  — best layer per word

Required data
-------------
H5 activation files in results/activations/{word}/{safe_model}/
All 8 models, all available words (currently: bank, bark, bat, crane).
"""

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from experiments.adequacy import (
    adequacy_margin,
    adequacy_best_layer,
    gdv_best_layer,
    layer_adequacy_profile,
    leave_one_out_adequacy_margins,
    normalized_adequacy_margin,
    save_profile_csv,
)

logger = logging.getLogger(__name__)

RESULTS_DIR   = "results"
OUTPUT_BASE   = Path("results/study/H1")
DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "light", "pitch"]


def _score_outer_sentence(X_train, labels_train, h_test, label_test):
    sense = int(label_test)
    wrong = 1 - sense
    c_correct = X_train[labels_train == sense].mean(axis=0)
    c_wrong = X_train[labels_train == wrong].mean(axis=0)
    return (
        adequacy_margin(h_test, c_correct, c_wrong),
        normalized_adequacy_margin(h_test, c_correct, c_wrong),
    )


def _nested_loo_best_vs_last(
    results_dir: str,
    model_name: str,
    word: str,
    epsilon: float,
):
    """Select a layer inside each fold, then score its untouched sentence."""
    safe_model = model_name.replace("/", "_")
    h5_dir = Path(results_dir) / "activations" / word / safe_model
    h5_files = sorted(
        h5_dir.glob("layer_*.h5"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    arrays = {}
    labels = None
    for path in h5_files:
        layer = int(path.stem.split("_")[1])
        with h5py.File(path, "r") as handle:
            arrays[layer] = handle["X"][:]
            current_labels = handle["labels"][:]
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise ValueError(f"Label order differs across layers for {model_name}/{word}")

    candidates = sorted(layer for layer in arrays if layer != 0)
    if not candidates or labels is None:
        return []
    last_layer = max(arrays)
    rows = []
    for held_out_idx in range(len(labels)):
        train_mask = np.arange(len(labels)) != held_out_idx
        labels_train = labels[train_mask]
        layer_scores = {}
        for layer in candidates:
            inner_raw, inner_norm = leave_one_out_adequacy_margins(
                arrays[layer][train_mask], labels_train
            )
            layer_scores[layer] = (
                float((inner_raw > epsilon).mean()),
                float(inner_norm.mean()),
            )
        selected_layer = max(candidates, key=lambda layer: layer_scores[layer])

        selected_raw, selected_norm = _score_outer_sentence(
            arrays[selected_layer][train_mask],
            labels_train,
            arrays[selected_layer][held_out_idx],
            labels[held_out_idx],
        )
        last_raw, last_norm = _score_outer_sentence(
            arrays[last_layer][train_mask],
            labels_train,
            arrays[last_layer][held_out_idx],
            labels[held_out_idx],
        )
        rows.append({
            "held_out_sentence_index": held_out_idx,
            "sense": int(labels[held_out_idx]),
            "selected_layer": selected_layer,
            "last_layer": last_layer,
            "M_raw_selected": round(float(selected_raw), 6),
            "M_norm_selected": round(float(selected_norm), 6),
            "adequate_selected": bool(selected_raw > epsilon),
            "M_raw_last": round(float(last_raw), 6),
            "M_norm_last": round(float(last_norm), 6),
            "adequate_last": bool(last_raw > epsilon),
        })
    return rows


def _available_words(results_dir: str, model_name: str, candidate_words: List[str]) -> List[str]:
    safe = model_name.replace("/", "_")
    available = []
    for w in candidate_words:
        h5_dir = Path(results_dir) / "activations" / w / safe
        if h5_dir.exists() and any(h5_dir.glob("layer_*.h5")):
            available.append(w)
        else:
            logger.warning("No complete H5 cache for model=%s word=%s - skipping.", model_name, w)
    return available


def run_h1(
    model_names: List[str],
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    epsilon: float = 0.0,
) -> None:
    """
    Run H1 for all given models and words.

    Parameters
    ----------
    model_names : models to analyse
    words       : homonyms to include (defaults to all with available H5 files)
    results_dir : root results directory containing activations/
    epsilon     : adequacy threshold (default 0 = correct side of boundary)
    """
    words = words or DEFAULT_WORDS
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        safe_model  = model_name.replace("/", "_")
        model_out   = OUTPUT_BASE / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        avail_words = _available_words(results_dir, model_name, words)
        if not avail_words:
            logger.error("No available words for %s - skipping H1.", model_name)
            continue

        summary_rows = []
        for word in avail_words:
            logger.info("[H1] %s / %s", model_name, word)

            profile = layer_adequacy_profile(
                results_dir, model_name, word,
                correct_sense=0, epsilon=epsilon,
                centroid_mode="leave_one_out",
            )
            if not profile:
                logger.warning("Empty profile for %s / %s", model_name, word)
                continue

            csv_path = model_out / f"h1_{word}.csv"
            save_profile_csv(profile, str(csv_path))

            nested_rows = _nested_loo_best_vs_last(
                results_dir, model_name, word, epsilon
            )
            if not nested_rows:
                logger.warning("No nested-LOO rows for %s / %s", model_name, word)
                continue
            nested_path = model_out / f"h1_nested_loo_{word}.csv"
            with open(nested_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=nested_rows[0].keys())
                writer.writeheader()
                writer.writerows(nested_rows)

            best_layer  = adequacy_best_layer(profile)
            last_layer  = max(profile)
            n_layers    = len(profile)

            # GDV best layer for comparison (if CSV exists)
            gdv_csv = Path(results_dir) / f"{safe_model}_gdv" / f"gdv_values_{word}.csv"
            gdv_best = gdv_best_layer(str(gdv_csv)) if gdv_csv.exists() else None

            summary_rows.append({
                "word":              word,
                "n_layers":          n_layers,
                "best_layer_M":      best_layer,
                "best_layer_selection_basis": "full_profile_leave_one_out",
                "best_layer_depth":  round(best_layer / max(last_layer, 1), 3),
                "mean_M_best_raw":   round(profile[best_layer]["mean_raw"], 4),
                "mean_M_last_raw":   round(profile[last_layer]["mean_raw"], 4),
                "mean_M_best_norm":  round(profile[best_layer]["mean_norm"], 4),
                "mean_M_last_norm":  round(profile[last_layer]["mean_norm"], 4),
                "frac_adeq_best":    round(profile[best_layer]["fraction_adequate"], 3),
                "frac_adeq_last":    round(profile[last_layer]["fraction_adequate"], 3),
                "nested_frac_selected": round(float(np.mean([
                    row["adequate_selected"] for row in nested_rows
                ])), 4),
                "nested_frac_last": round(float(np.mean([
                    row["adequate_last"] for row in nested_rows
                ])), 4),
                "nested_mean_M_norm_selected": round(float(np.mean([
                    row["M_norm_selected"] for row in nested_rows
                ])), 6),
                "nested_mean_M_norm_last": round(float(np.mean([
                    row["M_norm_last"] for row in nested_rows
                ])), 6),
                "nested_modal_selected_layer": Counter(
                    row["selected_layer"] for row in nested_rows
                ).most_common(1)[0][0],
                "nested_n_outer_folds": len(nested_rows),
                "gdv_best_layer":    gdv_best,
                "layers_agree":      best_layer == gdv_best if gdv_best is not None else None,
            })

        # Write summary
        if summary_rows:
            summary_path = model_out / "h1_summary.csv"
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)
            logger.info("[H1] Summary saved to %s", summary_path)
