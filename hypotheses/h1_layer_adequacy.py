"""
H1 — Layer Adequacy
====================
Hypothesis: Word-sense adequacy (M_l) differs across layers; the last layer is
not always the most adequate.

Method
------
For each (model, word) with cached activations:
  1. Load H5 activation cache.
  2. Compute M_l per layer per sentence (in-sample: centroids from full set).
  3. Report mean margin and fraction-adequate per layer.

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
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.adequacy import (
    adequacy_best_layer,
    gdv_best_layer,
    layer_adequacy_profile,
    save_profile_csv,
)

logger = logging.getLogger(__name__)

RESULTS_DIR   = "results"
OUTPUT_BASE   = Path("results/study/H1")
DEFAULT_WORDS = ["bank", "bark", "bat", "crane"]


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
            )
            if not profile:
                logger.warning("Empty profile for %s / %s", model_name, word)
                continue

            csv_path = model_out / f"h1_{word}.csv"
            save_profile_csv(profile, str(csv_path))

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
                "best_layer_depth":  round(best_layer / max(last_layer, 1), 3),
                "mean_M_best":       round(profile[best_layer]["mean"], 4),
                "mean_M_last":       round(profile[last_layer]["mean"], 4),
                "frac_adeq_best":    round(profile[best_layer]["fraction_adequate"], 3),
                "frac_adeq_last":    round(profile[last_layer]["fraction_adequate"], 3),
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
