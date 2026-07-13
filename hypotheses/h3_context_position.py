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

results/study/H3/h3_chance_test.csv
  per (model, condition): exact binomial test of frac_adequate against
  chance (0.5), pooled across all words' sentences (~80/model/condition,
  not collapsed to one binary outcome per word). This directly tests the
  claims above — decoders at chance in R should fail to reject; L and
  encoder conditions should reject decisively. An earlier design used a
  per-word sign test (L-mean > R-mean, n=8 words/model) instead; dropped
  because with n=8 the test saturates at its floor p-value (0.0078) for
  every decoder model simultaneously — it can't distinguish models from
  each other and adds no information beyond what the aggregate table
  already shows directly.

Required data
-------------
- results/activations/{word}/{safe_model}/ (profiling centroids)
- data/paired_sentences.json (L/R paired sentences — see inspect_paired_sentences.ipynb)
- Models: all 8 (4 encoder + 4 decoder — see model_registry.ALL_MODELS). Full
  coverage strengthens "encoder vs. decoder" claims to 4-vs-4 rather than the
  earlier 2-vs-2 representative subset.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments.adequacy import symmetric_adequacy_margins, symmetric_normalized_adequacy_margins, load_centroids
from models import get_target_activations, is_decoder_only, load_model_and_tokenizer  # is_decoder_only used for arch_type label only
from model_registry import ALL_MODELS
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR       = "results"
PAIRED_DATA_PATH  = "data/paired_sentences.json"
OUTPUT_BASE       = Path("results/study/H3")
# H0/H3/H4/H5 originally ran on a 4-model representative subset (2 enc + 2 dec)
# since they need fresh forward passes over paired/garden-path stimuli, unlike
# H1/H2 which reuse cached activations. Now full 8-model coverage — see
# model_registry.py for the single source of truth on model names/aliases.
H3_MODELS         = ALL_MODELS


def _load_paired_sentences(path: str, word: str) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    Load paired sentences for a given word from the JSON dataset.

    Returns (sentences, conditions, sentence_ids, senses)
    where condition is 'L' or 'R' and sense is each item's own true sense
    (0 or 1) — paired_sentences.json is balanced across both senses per
    word/condition, so callers must score each sentence against its own
    true sense rather than a single fixed sense-0/1 pair.
    """
    with open(path) as f:
        data = json.load(f)
    if word not in data:
        raise KeyError(f"Word '{word}' not found in {path}. Add it via the paired sentences notebook.")
    sentences, conditions, ids, senses = [], [], [], []
    for item in data[word]:
        sentences.append(item["sentence"])
        conditions.append(item["condition"])
        ids.append(item.get("id", f"{word}_{len(ids)}"))
        senses.append(item["sense"])
    return sentences, conditions, ids, senses


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
        # Always target-token: we measure the homonym's own representation.
        # For decoders this is causal (left-context only) — which is exactly the
        # effect H3 tests. R-condition sentences will show poor adequacy for
        # decoders because the homonym sits at position ~2 with no useful left context.
        arch_type = "decoder" if is_decoder_only(model) else "encoder"
        logger.info("[H3] %s (%s, pooling=target)", model_name, arch_type)

        for word in words:
            try:
                sentences, conditions, sent_ids, senses = _load_paired_sentences(paired_data_path, word)
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

            c0 = centroids[layer_idx][0]
            c1 = centroids[layer_idx][1]

            # Forward pass — extract homonym-position representations
            targets = [word] * len(sentences)
            acts = get_target_activations(
                model, tokenizer,
                sentences, targets,
                batch_size=4,
                layer_indices=[layer_idx],
                pooling="target",
            )
            H = acts[layer_idx].numpy()
            # paired_sentences.json is balanced across both senses per word/condition,
            # so each sentence must be scored against its own true sense.
            senses_arr  = np.array(senses)
            margins     = symmetric_adequacy_margins(H, senses_arr, c0, c1)
            # Normalized by inter-centroid distance — comparable across
            # architectures/layers, unlike raw M_l (see adequacy.py note).
            margins_norm = symmetric_normalized_adequacy_margins(H, senses_arr, c0, c1)

            # Write per-sentence results
            csv_path = model_out / f"h3_{word}.csv"
            rows = []
            for sid, cond, margin, margin_norm in zip(sent_ids, conditions, margins, margins_norm):
                rows.append({
                    "sentence_id": sid,
                    "condition":   cond,
                    "M_l":         round(float(margin), 4),
                    "M_l_norm":    round(float(margin_norm), 4),
                    "adequate":    margin > epsilon,
                    "layer":       layer_idx,
                })
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            # Aggregate per condition
            for cond in ("L", "R"):
                cond_mask = [c == cond for c in conditions]
                cond_margins      = margins[cond_mask]
                cond_margins_norm = margins_norm[cond_mask]
                if len(cond_margins) == 0:
                    continue
                aggregate_rows.append({
                    "model":           safe_model,
                    "arch_type":       arch_type,
                    "word":            word,
                    "condition":       cond,
                    "n":               len(cond_margins),
                    "mean_M_l":        round(float(cond_margins.mean()), 4),
                    "mean_M_l_norm":   round(float(cond_margins_norm.mean()), 4),
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

        compute_chance_level_tests(results_dir)


def compute_chance_level_tests(results_dir: str = RESULTS_DIR) -> List[Dict]:
    """
    Exact binomial test of frac_adequate against chance (p=0.5), per
    (model, condition), pooling sentence counts across all words. Requires
    h3_aggregate.csv (run_h3() writes it, then calls this automatically).

    Uses the full per-sentence N per (model, condition) — roughly 80
    sentences (8 words x ~10 sentences/word/condition) — rather than
    collapsing to a per-word binary outcome, which is both much better
    powered and a direct test of what H3 actually claims: not "does L
    tend to beat R across words" but "is adequacy in this condition
    distinguishable from a coin flip at all."
    """
    from scipy.stats import binomtest

    agg_path = Path(results_dir) / "study" / "H3" / "h3_aggregate.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"{agg_path} not found — run run_h3() first.")

    pooled: Dict[Tuple[str, str], Dict[str, int]] = {}
    arch_by_model: Dict[str, str] = {}
    with open(agg_path) as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["condition"])
            n = int(row["n"])
            n_adequate = round(float(row["frac_adequate"]) * n)
            entry = pooled.setdefault(key, {"n": 0, "n_adequate": 0})
            entry["n"] += n
            entry["n_adequate"] += n_adequate
            arch_by_model[row["model"]] = row["arch_type"]

    results = []
    for (model, condition), counts in pooled.items():
        n, k = counts["n"], counts["n_adequate"]
        test = binomtest(k, n, 0.5, alternative="two-sided")
        results.append({
            "model":         model,
            "arch_type":     arch_by_model.get(model, ""),
            "condition":     condition,
            "n_sentences":   n,
            "n_adequate":    k,
            "frac_adequate": round(k / n, 3),
            "p_value":       test.pvalue if test.pvalue >= 1e-4 else float(f"{test.pvalue:.2e}"),
        })

    out_path = Path(results_dir) / "study" / "H3" / "h3_chance_test.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info("[H3] Chance-level test saved to %s", out_path)
    return results
