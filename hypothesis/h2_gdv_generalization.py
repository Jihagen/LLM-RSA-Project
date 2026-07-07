"""
H2 — GDV Generalisation
========================
Hypothesis: The GDV-best layer identified on a profiling set of words
predicts the best adequacy layer on a held-out word.

Method (leave-one-out across words)
-------------------------------------
For each held-out word w_test:
  1. Compute GDV-best layer from the aggregate GDV CSV (profiling = other words).
  2. Evaluate mean M_l on w_test at: GDV-best layer, last layer, adequacy-best layer.
  3. Compare the three strategies.

Output
------
results/study/H2/{safe_model}/h2_loo.csv   — one row per fold (held-out word)

Required data
-------------
H5 files + gdv_values_{word}.csv for all words (needs complete 8-word run).
Currently runnable with 4 words (bank, bark, bat, crane) as LOO folds.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.adequacy import (
    adequacy_best_layer,
    gdv_best_layer,
    layer_adequacy_profile,
    load_centroids,
    batch_adequacy_margins,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
OUTPUT_BASE = Path("results/study/H2")


def _profile_gdv_best_layer(
    results_dir: str,
    safe_model: str,
    profiling_words: List[str],
) -> int:
    """
    Aggregate GDV across profiling words and return the best layer.
    Strategy: average GDV per layer across words, pick most negative.
    """
    gdv_dir = Path(results_dir) / f"{safe_model}_gdv"
    layer_gdv_sums: Dict[int, List[float]] = {}
    for word in profiling_words:
        csv_path = gdv_dir / f"gdv_values_{word}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                l = int(row["Layer"])
                if l == 0:
                    continue  # non-contextual embedding layer, not a valid candidate
                layer_gdv_sums.setdefault(l, []).append(float(row["GDV"]))
    if not layer_gdv_sums:
        raise FileNotFoundError(f"No GDV CSVs found for model={safe_model!r}")
    # Most negative mean GDV across profiling words
    return min(layer_gdv_sums, key=lambda l: np.mean(layer_gdv_sums[l]))


def _evaluate_layer(
    results_dir: str,
    model_name: str,
    word: str,
    layer_idx: int,
    epsilon: float = 0.0,
) -> Dict:
    """Evaluate M_l at a specific layer for a held-out word."""
    centroids = load_centroids(results_dir, model_name, word)
    if layer_idx not in centroids:
        return {"mean_M": float("nan"), "frac_adequate": float("nan")}

    safe_model = model_name.replace("/", "_")
    import h5py
    h5_path = Path(results_dir) / "activations" / word / safe_model / f"layer_{layer_idx}.h5"
    with h5py.File(h5_path, "r") as f:
        X      = f["X"][:]
        labels = f["labels"][:]

    c_correct = centroids[layer_idx][0]
    c_wrong   = centroids[layer_idx][1]
    margins   = batch_adequacy_margins(X, c_correct, c_wrong)
    return {
        "mean_M":        float(margins.mean()),
        "frac_adequate": float((margins > epsilon).mean()),
    }


def run_h2(
    model_names: List[str],
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    epsilon: float = 0.0,
) -> None:
    """
    Run H2 leave-one-out GDV generalisation for all models.
    Requires GDV CSVs (gdv_values_{word}.csv) and H5 files for all words.
    """
    words = words or ["bank", "bark", "bat", "crane"]
    if len(words) < 2:
        logger.error("[H2] Need at least 2 words for leave-one-out.")
        return

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out  = OUTPUT_BASE / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        rows = []
        for held_out in words:
            profiling = [w for w in words if w != held_out]
            logger.info("[H2] %s | held-out=%s, profiling=%s", model_name, held_out, profiling)

            try:
                gdv_layer = _profile_gdv_best_layer(results_dir, safe_model, profiling)
            except FileNotFoundError as e:
                logger.warning("[H2] %s", e)
                continue

            # Adequacy profile on held-out word to find empirical best layer
            try:
                profile = layer_adequacy_profile(results_dir, model_name, held_out,
                                                 correct_sense=0, epsilon=epsilon)
            except FileNotFoundError as e:
                logger.warning("[H2] %s", e)
                continue

            if not profile:
                continue

            adeq_layer = adequacy_best_layer(profile)
            last_layer = max(profile)

            rows.append({
                "held_out_word":       held_out,
                "gdv_best_layer":      gdv_layer,
                "adeq_best_layer":     adeq_layer,
                "last_layer":          last_layer,
                "mean_M_gdv_best":     round(profile.get(gdv_layer,  {}).get("mean", float("nan")), 4),
                "mean_M_adeq_best":    round(profile[adeq_layer]["mean"], 4),
                "mean_M_last":         round(profile[last_layer]["mean"], 4),
                "frac_gdv_best":       round(profile.get(gdv_layer,  {}).get("fraction_adequate", float("nan")), 3),
                "frac_adeq_best":      round(profile[adeq_layer]["fraction_adequate"], 3),
                "frac_last":           round(profile[last_layer]["fraction_adequate"], 3),
                "gdv_matches_adeq":    gdv_layer == adeq_layer,
            })

        if rows:
            out_path = model_out / "h2_loo.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info("[H2] LOO results saved to %s", out_path)
