"""
H3 — Right-Context Vulnerability
==================================
Hypothesis: Decoder homonym-token representations are inadequate when
decisive context appears after the homonym (R condition); encoders are
largely unaffected by context position.

Method
------
For each (model, word):
  1. Load sense centroids from profiling H5 files.
  2. Run a forward pass on paired sentences (L = left-context biased,
     R = right-context biased) from data/paired_sentences.json.
  3. Extract the representation at the homonym-token position (target pooling).
  4. Compute M_l per sentence. Compare L vs R distributions.

Output
------
results/study/H3/{safe_model}/h3_{word}.csv
  columns: sentence_id, condition (L/R), M_l, adequate (bool)

results/study/H3/h3_aggregate.csv
  mean M_l and fraction adequate per (model, arch_type, condition)

Required data
-------------
- results/activations/{word}/{safe_model}/ (profiling centroids)
- data/paired_sentences.json (L/R paired sentences — see inspect_paired_sentences.ipynb)
- Models: DeBERTa-v3-large, RoBERTa-large, Mistral-Nemo-Base-2407, Qwen2.5-3B
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments.adequacy import batch_adequacy_margins, load_centroids
from models import get_target_activations, is_decoder_only, load_model_and_tokenizer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR       = "results"
PAIRED_DATA_PATH  = "data/paired_sentences.json"
OUTPUT_BASE       = Path("results/study/H3")
H3_MODELS         = [
    "microsoft/deberta-v3-large",
    "FacebookAI/roberta-large",
    "mistralai/Mistral-Nemo-Base-2407",
    "Qwen/Qwen2.5-3B",
]


def _load_paired_sentences(path: str, word: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load paired sentences for a given word from the JSON dataset.

    Returns (sentences, conditions, sentence_ids)
    where condition is 'L' or 'R'.
    """
    with open(path) as f:
        data = json.load(f)
    if word not in data:
        raise KeyError(f"Word '{word}' not found in {path}. Add it via the paired sentences notebook.")
    sentences, conditions, ids = [], [], []
    for item in data[word]:
        sentences.append(item["sentence"])
        conditions.append(item["condition"])
        ids.append(item.get("id", f"{word}_{len(ids)}"))
    return sentences, conditions, ids


def _select_layer(model_name: str, results_dir: str, word: str) -> int:
    """Pick the adequacy-best layer from the H1 summary if available, else last layer."""
    safe = model_name.replace("/", "_")
    h1_summary = Path(results_dir) / "study" / "H1" / safe / "h1_summary.csv"
    if h1_summary.exists():
        with open(h1_summary) as f:
            for row in csv.DictReader(f):
                if row["word"] == word:
                    return int(row["best_layer_M"])
    # fallback: count layers from H5 files
    h5_dir = Path(results_dir) / "activations" / word / safe
    if h5_dir.exists():
        layers = [int(p.stem.split("_")[1]) for p in h5_dir.glob("layer_*.h5")]
        if layers:
            return max(layers)
    raise FileNotFoundError(f"Cannot determine best layer for {model_name} / {word}")


def run_h3(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    paired_data_path: str = PAIRED_DATA_PATH,
    epsilon: float = 0.0,
) -> None:
    """
    Run H3 for specified models and words.
    Requires paired_sentences.json and H5 profiling files.
    """
    model_names = model_names or H3_MODELS
    words       = words or ["bank", "bark", "bat", "crane"]
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not Path(paired_data_path).exists():
        logger.error(
            "[H3] Paired sentence dataset not found at %s. "
            "Run data/inspect_paired_sentences.ipynb to create it.", paired_data_path
        )
        return

    aggregate_rows = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out  = OUTPUT_BASE / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        model, tokenizer = load_model_and_tokenizer(model_name)
        pooling   = "last_token" if is_decoder_only(model) else "target"
        arch_type = "decoder" if is_decoder_only(model) else "encoder"
        logger.info("[H3] %s (%s, pooling=%s)", model_name, arch_type, pooling)

        for word in words:
            try:
                sentences, conditions, sent_ids = _load_paired_sentences(paired_data_path, word)
            except KeyError as e:
                logger.warning("[H3] %s", e)
                continue

            try:
                centroids  = load_centroids(results_dir, model_name, word)
                layer_idx  = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as e:
                logger.warning("[H3] %s", e)
                continue

            if layer_idx not in centroids:
                logger.warning("[H3] Layer %d not in centroid cache for %s / %s", layer_idx, model_name, word)
                continue

            c_correct = centroids[layer_idx][0]
            c_wrong   = centroids[layer_idx][1]

            # Forward pass — extract homonym-position representations
            targets = [word] * len(sentences)
            acts = get_target_activations(
                model, tokenizer,
                sentences, targets,
                batch_size=4,
                layer_indices=[layer_idx],
                pooling=pooling,
            )
            H = acts[layer_idx].numpy()
            margins = batch_adequacy_margins(H, c_correct, c_wrong)

            # Write per-sentence results
            csv_path = model_out / f"h3_{word}.csv"
            rows = []
            for sid, cond, margin in zip(sent_ids, conditions, margins):
                rows.append({
                    "sentence_id": sid,
                    "condition":   cond,
                    "M_l":         round(float(margin), 4),
                    "adequate":    margin > epsilon,
                    "layer":       layer_idx,
                })
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            # Aggregate per condition
            for cond in ("L", "R"):
                cond_margins = margins[[c == cond for c in conditions]]
                if len(cond_margins) == 0:
                    continue
                aggregate_rows.append({
                    "model":           safe_model,
                    "arch_type":       arch_type,
                    "word":            word,
                    "condition":       cond,
                    "n":               len(cond_margins),
                    "mean_M_l":        round(float(cond_margins.mean()), 4),
                    "frac_adequate":   round(float((cond_margins > epsilon).mean()), 3),
                    "layer_used":      layer_idx,
                })

            logger.info("[H3] %s / %s | layer=%d | L_mean=%.3f R_mean=%.3f",
                        model_name, word, layer_idx,
                        margins[[c == "L" for c in conditions]].mean() if "L" in conditions else float("nan"),
                        margins[[c == "R" for c in conditions]].mean() if "R" in conditions else float("nan"))

    if aggregate_rows:
        agg_path = OUTPUT_BASE / "h3_aggregate.csv"
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)
        logger.info("[H3] Aggregate saved to %s", agg_path)
