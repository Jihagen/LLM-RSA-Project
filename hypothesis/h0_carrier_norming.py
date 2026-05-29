"""
H0 — Carrier Norming (prerequisite for H3/H4)
=============================================
Purpose
-------
Computes M_l at three levels for each (model, word):

  1. word_alone_M_l  — bare word ("bank") as its own sentence.
                       Should be near zero; any deviation reveals an intrinsic
                       lexical embedding bias before any sentence context.

  2. carrier_M_l     — word inside the ambiguous carrier sentence
                       ("The bank was unstable."), no disambiguating clause.
                       Shows what the sentence frame alone contributes.

  3. context_gain    — M_l(full sentence) - M_l(carrier)   [computed in H3]
                       Shows what the context clause specifically adds.

Output
------
results/study/H0/{safe_model}/h0_{word}.csv
  columns: carrier, sense, M_l_word_alone, M_l_carrier,
           carrier_shift (carrier - word_alone), biased (|M_l_carrier| > 0.3)

results/study/H0/h0_summary.csv
  per (model, word): mean |M_l_word_alone|, mean |M_l_carrier|, mean carrier_shift

Required data
-------------
- data/paired_sentences.json (must contain 'carrier' field on each item)
- results/activations/{word}/{safe_model}/ (centroids)
- Models: all 4 H3 models
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from experiments.adequacy import (
    compute_carrier_margins,
    compute_word_alone_margins,
    load_centroids,
)
from hypothesis.h3_context_position import H3_MODELS, PAIRED_DATA_PATH, _select_layer
from models import load_model_and_tokenizer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
OUTPUT_BASE = Path("results/study/H0")


def _extract_unique_carriers(paired_data: dict, word: str) -> List[dict]:
    """Extract one record per unique (carrier, sense) pair for a word."""
    seen, records = set(), []
    for item in paired_data.get(word, []):
        if "carrier" not in item:
            logger.warning("[H0] Item %s missing 'carrier' field.", item.get("id"))
            continue
        key = (item["carrier"], item["sense"])
        if key not in seen:
            seen.add(key)
            records.append({"word": word, "carrier": item["carrier"], "sense": item["sense"]})
    return records


def run_h0(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    paired_data_path: str = PAIRED_DATA_PATH,
) -> None:
    """
    Run H0 carrier norming for all specified models and words.
    Results are saved to results/study/H0/ and used by H3 for context_gain.
    """
    model_names = model_names or H3_MODELS
    words       = words or ["bank", "bark", "bat", "crane", "spring", "match", "light", "pitch"]
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not Path(paired_data_path).exists():
        logger.error("[H0] Paired data not found at %s.", paired_data_path)
        return

    with open(paired_data_path) as f:
        paired_data = json.load(f)

    summary_rows = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out  = OUTPUT_BASE / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        model, tokenizer = load_model_and_tokenizer(model_name)
        logger.info("[H0] %s", model_name)

        for word in words:
            carrier_items = _extract_unique_carriers(paired_data, word)
            if not carrier_items:
                logger.warning("[H0] No carriers found for word '%s'.", word)
                continue

            try:
                centroids = load_centroids(results_dir, model_name, word)
                layer_idx = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as e:
                logger.warning("[H0] %s", e)
                continue

            # ── Level 1: word alone ───────────────────────────────────────
            word_alone = compute_word_alone_margins(
                model, tokenizer, [word], centroids, layer_idx,
            )
            m_word_alone = word_alone.get(word, float("nan"))

            # ── Level 2: carrier sentences ────────────────────────────────
            records = compute_carrier_margins(
                model, tokenizer, carrier_items, centroids, layer_idx,
            )
            if not records:
                continue

            # Enrich each carrier record with the word-alone baseline
            for r in records:
                r["M_l_word_alone"]  = round(m_word_alone, 4)
                r["carrier_shift"]   = round(r["M_l_carrier"] - m_word_alone, 4)

            csv_path = model_out / f"h0_{word}.csv"
            fieldnames = ["word", "carrier", "sense",
                          "M_l_word_alone", "M_l_carrier", "carrier_shift",
                          "biased", "layer"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)

            carrier_margins = [abs(r["M_l_carrier"]) for r in records]
            carrier_shifts  = [abs(r["carrier_shift"]) for r in records]
            n_biased = sum(1 for r in records if r["biased"])
            summary_rows.append({
                "model":                 safe_model,
                "word":                  word,
                "n_carriers":            len(records),
                "mean_abs_M_l_word":     round(abs(m_word_alone), 4),
                "mean_abs_M_l_carrier":  round(np.mean(carrier_margins), 4),
                "mean_abs_carrier_shift":round(np.mean(carrier_shifts), 4),
                "n_biased":              n_biased,
                "layer_used":            layer_idx,
            })
            logger.info(
                "[H0] %s / %s | word_alone=%.3f carrier=%.3f shift=%.3f biased=%d",
                model_name, word,
                abs(m_word_alone), np.mean(carrier_margins),
                np.mean(carrier_shifts), n_biased,
            )

    if summary_rows:
        summary_path = OUTPUT_BASE / "h0_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("[H0] Summary saved to %s", summary_path)
