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
  2. Compute M_l for the final position against TWO centroid pairs, same
     rationale as H4: primary uses homonym-position centroids (consistent
     with every other hypothesis's definition of "adequate" / "resolved");
     secondary uses final-position centroids from the same profiling pass
     (see gdv_experiments.py), a local-recoverability diagnostic. target_h
     is only ever scored against homonym-position centroids (no other
     geometry exists for that token). Both margins are also reported
     normalized by their own inter-centroid distance.
  3. Report: at what position, and under which geometry, does the
     representation shift from the primed (wrong) sense toward the resolved
     (correct) sense?

For encoders: shift should be visible across layers (early layers commit to
primed sense; late layers correct to resolved sense).
For decoders: homonym-token is stuck with primed sense; final token may shift.

This dual scoring exists specifically to distinguish two different reasons
a low primary priming-conflict rate could occur: (a) the resolution is
there, just expressed in the final position's own geometry rather than the
homonym's (as found for H4) — the secondary score should then be
substantially higher; vs (b) the representation genuinely does not commit
to either sense at the final position — in which case the secondary score
should stay low too, and per-sentence margins (both geometries) should
cluster near zero rather than being consistently negative.

Three-way resolution bucketing
-------------------------------
The binary priming_conflict flag (target inadequate AND final adequate)
collapses two very different outcomes into one "not conflict" bucket:
a final-position representation that sits near the decision boundary
(genuinely undecided) and one that sits confidently on the WRONG side
(decisively stuck on the primed sense, i.e. failed revision, not
indecision). These are mechanistically different and worth separating.

Each sentence's final-position normalized margin (both geometries) is
bucketed into:
  - "confident_correct" : margin >  CONFIDENCE_THRESHOLD
  - "confident_primed"  : margin < -CONFIDENCE_THRESHOLD
  - "ambiguous"          : |margin| <= CONFIDENCE_THRESHOLD (near the
                           boundary — not confidently either sense)
CONFIDENCE_THRESHOLD = 0.3, i.e. margin covers at least 30% of the
distance from the boundary to a centroid. This is a round-number
heuristic, not fit to the data; results should be checked for sensitivity
to this choice before treating exact bucket percentages as precise.

Output
------
results/study/H5/{safe_model}/h5_{word}.csv
  columns: sentence_id, M_l_target_correct, M_l_final_correct,
           M_l_final_correct_localcentroid, *_norm variants,
           priming_conflict, priming_conflict_localcentroid,
           resolution_bucket, resolution_bucket_localcentroid

results/study/H5/h5_aggregate.csv
  includes std_M_final_norm / std_M_final_localcentroid_norm per cell (to
  help separate a genuine near-zero/ambiguous signature from a noisy,
  high-variance, too-little-data signature) and per-cell fractions of each
  of the three resolution buckets, both geometries.

Required data
-------------
- data/garden_path_sentences.json (6-7 sentences/word, up from 2-3)
- results/activations/{word}/{safe_model}/ (homonym-position centroids)
- results/activations_final/{word}/{safe_model}/ (final-position centroids)

Note
----
This experiment is EXPLORATORY. Garden-path sentences have two sense labels:
  - primed_sense: the sense implied by left context
  - correct_sense: the sense intended by the full sentence

The JSON format must include both labels per sentence (see notebook).

Data-quality fix (this expansion round): the "spring" and "pitch" entries
previously used a sense that did NOT match either of the two senses the
profiling data / centroids were actually built on — "spring" revealed a
mechanical coiled spring instead of the trained water-source sense, and
"pitch" primed a musical-tone sense instead of the trained business/sales
sense. Both were scoring against effectively unrelated centroids. Fixed to
use the correct sense pair, verified against data/synthetic_data_h2.pkl.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.adequacy import (
    batch_adequacy_margins,
    load_centroids,
    load_final_centroids,
    normalized_adequacy_margin,
)
from hypotheses.h3_context_position import H3_MODELS, _select_layer
from models import get_dual_position_activations, is_decoder_only, load_model_and_tokenizer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR      = "results"
GP_DATA_PATH     = "data/garden_path_sentences.json"
OUTPUT_BASE      = Path("results/study/H5")

# See module docstring, "Three-way resolution bucketing".
CONFIDENCE_THRESHOLD = 0.3


def _resolution_bucket(margin_norm: float, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    if margin_norm > threshold:
        return "confident_correct"
    if margin_norm < -threshold:
        return "confident_primed"
    return "ambiguous"


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
                centroids       = load_centroids(results_dir, model_name, word)
                final_centroids = load_final_centroids(results_dir, model_name, word)
                layer_idx       = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as e:
                logger.warning("[H5] %s", e)
                continue

            if layer_idx not in centroids or layer_idx not in final_centroids:
                continue

            target_acts, final_acts = get_dual_position_activations(
                model, tokenizer, sentences, targets,
                batch_size=4, layer_indices=[layer_idx],
            )
            H_target = target_acts[layer_idx].numpy()
            H_final  = final_acts[layer_idx].numpy()

            csv_rows = []
            n_conflict = 0
            n_conflict_local = 0
            for i, (sid, c_sense, p_sense) in enumerate(zip(sent_ids, correct_ids, primed_ids)):
                if c_sense not in centroids[layer_idx] or p_sense not in centroids[layer_idx]:
                    continue
                if c_sense not in final_centroids[layer_idx] or p_sense not in final_centroids[layer_idx]:
                    continue

                c_correct  = centroids[layer_idx][c_sense]
                c_primed   = centroids[layer_idx][p_sense]
                fc_correct = final_centroids[layer_idx][c_sense]
                fc_primed  = final_centroids[layer_idx][p_sense]

                mt_correct       = adequacy_margin_single(H_target[i], c_correct, c_primed)
                mf_correct       = adequacy_margin_single(H_final[i],  c_correct, c_primed)        # PRIMARY: homonym geometry
                mf_correct_local = adequacy_margin_single(H_final[i],  fc_correct, fc_primed)       # SECONDARY: final-position geometry

                mt_correct_norm       = normalized_adequacy_margin(H_target[i], c_correct, c_primed)
                mf_correct_norm       = normalized_adequacy_margin(H_final[i],  c_correct, c_primed)
                mf_correct_local_norm = normalized_adequacy_margin(H_final[i],  fc_correct, fc_primed)

                # Priming conflict: target aligns with primed sense, final aligns with correct
                conflict       = (mt_correct < epsilon) and (mf_correct       > epsilon)
                conflict_local = (mt_correct < epsilon) and (mf_correct_local > epsilon)
                if conflict:
                    n_conflict += 1
                if conflict_local:
                    n_conflict_local += 1

                bucket       = _resolution_bucket(mf_correct_norm)
                bucket_local = _resolution_bucket(mf_correct_local_norm)

                csv_rows.append({
                    "sentence_id":                      sid,
                    "correct_sense":                    c_sense,
                    "primed_sense":                     p_sense,
                    "M_l_target_correct":                round(float(mt_correct), 4),
                    "M_l_final_correct":                 round(float(mf_correct), 4),
                    "M_l_final_correct_localcentroid":   round(float(mf_correct_local), 4),
                    "M_l_target_correct_norm":           round(float(mt_correct_norm), 4),
                    "M_l_final_correct_norm":            round(float(mf_correct_norm), 4),
                    "M_l_final_correct_localcentroid_norm": round(float(mf_correct_local_norm), 4),
                    "priming_conflict":                  conflict,
                    "priming_conflict_localcentroid":    conflict_local,
                    "resolution_bucket":                 bucket,
                    "resolution_bucket_localcentroid":   bucket_local,
                    "layer":                             layer_idx,
                })

            if not csv_rows:
                continue

            csv_path = model_out / f"h5_{word}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

            n = len(csv_rows)
            final_norms       = [r["M_l_final_correct_norm"]                for r in csv_rows]
            final_local_norms = [r["M_l_final_correct_localcentroid_norm"]  for r in csv_rows]
            buckets       = [r["resolution_bucket"]              for r in csv_rows]
            buckets_local = [r["resolution_bucket_localcentroid"] for r in csv_rows]
            aggregate_rows.append({
                "model":              safe_model,
                "arch_type":          arch_type,
                "word":               word,
                "n_gp_sentences":     n,
                "priming_conflict_rate":              round(n_conflict / n, 3),
                "priming_conflict_rate_localcentroid": round(n_conflict_local / n, 3),
                "mean_M_target":      round(np.mean([r["M_l_target_correct"] for r in csv_rows]), 4),
                "mean_M_final":       round(np.mean([r["M_l_final_correct"]  for r in csv_rows]), 4),
                "mean_M_final_localcentroid": round(np.mean([r["M_l_final_correct_localcentroid"] for r in csv_rows]), 4),
                "mean_M_final_norm":          round(float(np.mean(final_norms)), 4),
                "mean_M_final_localcentroid_norm": round(float(np.mean(final_local_norms)), 4),
                # Spread of the per-sentence normalized margins — low std + mean near
                # zero suggests genuine unresolved ambiguity (hypothesis b in the
                # accompanying discussion); high std suggests a noisy/underpowered
                # signal (hypothesis c) rather than a consistent one either way.
                "std_M_final_norm":               round(float(np.std(final_norms)), 4),
                "std_M_final_localcentroid_norm":  round(float(np.std(final_local_norms)), 4),
                # Three-way resolution bucket fractions (see module docstring).
                # local-centroid is the theoretically appropriate geometry;
                # homonym-centroid kept for direct comparability with H4.
                "frac_confident_correct":              round(buckets.count("confident_correct") / n, 3),
                "frac_confident_primed":                round(buckets.count("confident_primed") / n, 3),
                "frac_ambiguous":                       round(buckets.count("ambiguous") / n, 3),
                "frac_confident_correct_localcentroid": round(buckets_local.count("confident_correct") / n, 3),
                "frac_confident_primed_localcentroid":  round(buckets_local.count("confident_primed") / n, 3),
                "frac_ambiguous_localcentroid":          round(buckets_local.count("ambiguous") / n, 3),
                "layer_used":         layer_idx,
            })
            logger.info(
                "[H5] %s / %s | conflict_rate=%d/%d (local=%d/%d)",
                model_name, word, n_conflict, n, n_conflict_local, n,
            )

    if aggregate_rows:
        agg_path = OUTPUT_BASE / "h5_aggregate.csv"
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)
        logger.info("[H5] Aggregate saved to %s", agg_path)


def adequacy_margin_single(h: np.ndarray, c_correct: np.ndarray, c_wrong: np.ndarray) -> float:
    return float(np.linalg.norm(h - c_wrong) - np.linalg.norm(h - c_correct))
