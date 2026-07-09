"""
H4 — Decoder Recovery / Token-Position Dissociation
=====================================================
Hypothesis: In decoders, the homonym-token representation may be inadequate
while a later (final) token representation reflects the resolved global meaning.

Method
------
For R-condition sentences (where disambiguating context follows the homonym):
  1. Run a single forward pass with get_dual_position_activations.
  2. Extract two representations per sentence, per layer:
     - target_h : mean-pooled homonym-token hidden state
     - final_h  : last non-padding token hidden state
  3. Compute M_l at both positions using the same centroids.
  4. Test: target_M_l < epsilon AND final_M_l > epsilon (dissociation).

Output
------
results/study/H4/{safe_model}/h4_{word}.csv
  columns: sentence_id, M_l_target, M_l_final, adequate_target, adequate_final, dissociation

results/study/H4/h4_aggregate.csv
  dissociation rate per (model, arch_type, word)

Required data
-------------
- data/paired_sentences.json (R-condition sentences)
- results/activations/{word}/{safe_model}/ (profiling centroids)
- Models: same 4 as H3

Note on encoder behaviour
--------------------------
For encoders, target_h already aggregates full bidirectional context, so
target_M_l ≈ final_M_l is expected. Any dissociation would be surprising.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.adequacy import symmetric_adequacy_margins, load_centroids
from hypotheses.h3_context_position import H3_MODELS, PAIRED_DATA_PATH, _select_layer
from models import get_dual_position_activations, is_decoder_only, load_model_and_tokenizer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
OUTPUT_BASE = Path("results/study/H4")


def run_h4(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    paired_data_path: str = PAIRED_DATA_PATH,
    epsilon: float = 0.0,
) -> None:
    """
    Run H4 dissociation analysis for specified models and words.
    Uses R-condition sentences from paired_sentences.json.
    """
    model_names = model_names or H3_MODELS
    words       = words or ["bank", "bark", "bat", "crane"]
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not Path(paired_data_path).exists():
        logger.error("[H4] Paired sentence dataset not found at %s.", paired_data_path)
        return

    with open(paired_data_path) as f:
        paired_data = json.load(f)

    aggregate_rows = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out  = OUTPUT_BASE / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        model, tokenizer = load_model_and_tokenizer(model_name)
        arch_type = "decoder" if is_decoder_only(model) else "encoder"
        logger.info("[H4] %s (%s)", model_name, arch_type)

        for word in words:
            if word not in paired_data:
                logger.warning("[H4] Word '%s' not in paired data - skipping.", word)
                continue

            # Keep only R-condition sentences
            r_items = [item for item in paired_data[word] if item["condition"] == "R"]
            if not r_items:
                logger.warning("[H4] No R-condition sentences for word '%s'.", word)
                continue

            sentences = [item["sentence"] for item in r_items]
            sent_ids  = [item.get("id", f"{word}_R_{i}") for i, item in enumerate(r_items)]
            senses    = [item["sense"] for item in r_items]
            targets   = [word] * len(sentences)

            try:
                centroids = load_centroids(results_dir, model_name, word)
                layer_idx = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as e:
                logger.warning("[H4] %s", e)
                continue

            if layer_idx not in centroids:
                continue

            c0 = centroids[layer_idx][0]
            c1 = centroids[layer_idx][1]

            target_acts, final_acts = get_dual_position_activations(
                model, tokenizer, sentences, targets,
                batch_size=4, layer_indices=[layer_idx],
            )
            H_target = target_acts[layer_idx].numpy()
            H_final  = final_acts[layer_idx].numpy()

            # R-condition sentences are balanced across both senses, so each
            # sentence must be scored against its own true sense.
            senses_arr = np.array(senses)
            m_target = symmetric_adequacy_margins(H_target, senses_arr, c0, c1)
            m_final  = symmetric_adequacy_margins(H_final,  senses_arr, c0, c1)

            csv_rows = []
            n_dissociation = 0
            for sid, mt, mf in zip(sent_ids, m_target, m_final):
                adeq_t = bool(mt > epsilon)
                adeq_f = bool(mf > epsilon)
                dissoc = (not adeq_t) and adeq_f  # inadequate at homonym, adequate at final
                if dissoc:
                    n_dissociation += 1
                csv_rows.append({
                    "sentence_id":     sid,
                    "M_l_target":      round(float(mt), 4),
                    "M_l_final":       round(float(mf), 4),
                    "adequate_target": adeq_t,
                    "adequate_final":  adeq_f,
                    "dissociation":    dissoc,
                    "layer":           layer_idx,
                })

            csv_path = model_out / f"h4_{word}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

            n = len(csv_rows)
            aggregate_rows.append({
                "model":              safe_model,
                "arch_type":          arch_type,
                "word":               word,
                "n_R_sentences":      n,
                "mean_M_target":      round(float(m_target.mean()), 4),
                "mean_M_final":       round(float(m_final.mean()), 4),
                "frac_adeq_target":   round(float((m_target > epsilon).mean()), 3),
                "frac_adeq_final":    round(float((m_final  > epsilon).mean()), 3),
                "dissociation_rate":  round(n_dissociation / n, 3),
                "layer_used":         layer_idx,
            })
            logger.info("[H4] %s / %s | dissociation=%d/%d | target_mean=%.3f final_mean=%.3f",
                        model_name, word, n_dissociation, n, m_target.mean(), m_final.mean())

    if aggregate_rows:
        agg_path = OUTPUT_BASE / "h4_aggregate.csv"
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)
        logger.info("[H4] Aggregate saved to %s", agg_path)
