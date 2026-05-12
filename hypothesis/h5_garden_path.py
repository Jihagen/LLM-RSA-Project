"""
H5 — Garden-Path / Representational Revision (Exploratory)
===========================================================
Hypothesis: Sentences where strong left context primes the wrong sense
(garden-path style) produce a measurable representational conflict that is
visible across layers or token positions.

Method
------
For garden-path sentences (initial context primes sense A; resolving context
forces sense B):
  1. Use get_dual_position_activations to extract homonym-position and
     final-position representations in one pass.
  2. Compute M_l at both positions against BOTH sense centroids.
  3. Report: at what position does the representation shift from the primed
     (wrong) sense toward the resolved (correct) sense?

For encoders: shift should be visible across layers (early layers commit to
primed sense; late layers correct to resolved sense).
For decoders: homonym-token is stuck with primed sense; final token may shift.

Output
------
results/study/H5/{safe_model}/h5_{word}.csv
  columns: sentence_id, M_l_target_correct, M_l_target_primed,
           M_l_final_correct, M_l_final_primed, priming_conflict

results/study/H5/h5_aggregate.csv

Required data
-------------
- data/garden_path_sentences.json
- results/activations/{word}/{safe_model}/ (centroids)

Note
----
This experiment is EXPLORATORY. Garden-path sentences have two sense labels:
  - primed_sense: the sense implied by left context
  - correct_sense: the sense intended by the full sentence

The JSON format must include both labels per sentence (see notebook).
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.adequacy import batch_adequacy_margins, load_centroids
from hypothesis.h3_context_position import H3_MODELS, _select_layer
from models import get_dual_position_activations, is_decoder_only, load_model_and_tokenizer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR      = "results"
GP_DATA_PATH     = "data/garden_path_sentences.json"
OUTPUT_BASE      = Path("results/study/H5")


def _load_garden_path(path: str, word: str) -> List[Dict]:
    """
    Load garden-path sentences for a word.

    Expected JSON structure:
    {
      "bank": [
        {
          "id": "bank_gp_01",
          "sentence": "...",
          "word": "bank",
          "primed_sense": 1,    // sense implied by left context
          "correct_sense": 0    // sense intended by full sentence
        }, ...
      ]
    }
    """
    with open(path) as f:
        data = json.load(f)
    if word not in data:
        raise KeyError(f"Word '{word}' not in {path}")
    return data[word]


def run_h5(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    gp_data_path: str = GP_DATA_PATH,
    epsilon: float = 0.0,
) -> None:
    """
    Run H5 garden-path analysis (exploratory).
    Requires garden_path_sentences.json (see data/ directory).
    """
    model_names = model_names or H3_MODELS
    words       = words or ["bank", "bark", "bat", "crane"]
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not Path(gp_data_path).exists():
        logger.warning(
            "[H5] Garden-path dataset not found at %s. "
            "Create it via data/inspect_paired_sentences.ipynb.", gp_data_path
        )
        return

    with open(gp_data_path) as f:
        gp_data = json.load(f)

    aggregate_rows = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out  = OUTPUT_BASE / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        model, tokenizer = load_model_and_tokenizer(model_name)
        arch_type = "decoder" if is_decoder_only(model) else "encoder"
        logger.info("[H5] %s (%s)", model_name, arch_type)

        for word in words:
            try:
                items = _load_garden_path(gp_data_path, word)
            except KeyError as e:
                logger.warning("[H5] %s", e)
                continue

            sentences    = [item["sentence"]      for item in items]
            sent_ids     = [item.get("id", f"{word}_gp_{i}") for i, item in enumerate(items)]
            correct_ids  = [item["correct_sense"] for item in items]
            primed_ids   = [item["primed_sense"]  for item in items]
            targets      = [word] * len(sentences)

            try:
                centroids = load_centroids(results_dir, model_name, word)
                layer_idx = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as e:
                logger.warning("[H5] %s", e)
                continue

            if layer_idx not in centroids:
                continue

            target_acts, final_acts = get_dual_position_activations(
                model, tokenizer, sentences, targets,
                batch_size=4, layer_indices=[layer_idx],
            )
            H_target = target_acts[layer_idx].numpy()
            H_final  = final_acts[layer_idx].numpy()

            csv_rows = []
            n_conflict = 0
            for i, (sid, c_sense, p_sense) in enumerate(zip(sent_ids, correct_ids, primed_ids)):
                if c_sense not in centroids[layer_idx] or p_sense not in centroids[layer_idx]:
                    continue

                c_correct = centroids[layer_idx][c_sense]
                c_primed  = centroids[layer_idx][p_sense]

                mt_correct = adequacy_margin_single(H_target[i], c_correct, c_primed)
                mf_correct = adequacy_margin_single(H_final[i],  c_correct, c_primed)

                # Priming conflict: target aligns with primed sense, final aligns with correct
                conflict = (mt_correct < epsilon) and (mf_correct > epsilon)
                if conflict:
                    n_conflict += 1

                csv_rows.append({
                    "sentence_id":          sid,
                    "correct_sense":        c_sense,
                    "primed_sense":         p_sense,
                    "M_l_target_correct":   round(float(mt_correct), 4),
                    "M_l_final_correct":    round(float(mf_correct), 4),
                    "priming_conflict":     conflict,
                    "layer":                layer_idx,
                })

            if not csv_rows:
                continue

            csv_path = model_out / f"h5_{word}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

            n = len(csv_rows)
            aggregate_rows.append({
                "model":              safe_model,
                "arch_type":          arch_type,
                "word":               word,
                "n_gp_sentences":     n,
                "priming_conflict_rate": round(n_conflict / n, 3),
                "mean_M_target":      round(np.mean([r["M_l_target_correct"] for r in csv_rows]), 4),
                "mean_M_final":       round(np.mean([r["M_l_final_correct"]  for r in csv_rows]), 4),
                "layer_used":         layer_idx,
            })
            logger.info("[H5] %s / %s | conflict_rate=%d/%d", model_name, word, n_conflict, n)

    if aggregate_rows:
        agg_path = OUTPUT_BASE / "h5_aggregate.csv"
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)
        logger.info("[H5] Aggregate saved to %s", agg_path)


def adequacy_margin_single(h: np.ndarray, c_correct: np.ndarray, c_wrong: np.ndarray) -> float:
    return float(np.linalg.norm(h - c_wrong) - np.linalg.norm(h - c_correct))
