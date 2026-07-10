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
     - final_h  : last non-padding, non-special-token hidden state
  3. Compute M_l for final_h against TWO centroid pairs, reported side by side:
     - homonym-position centroids (load_centroids) — the PRIMARY score,
       consistent with every other hypothesis's definition of "adequate":
       does final_h sit closer to the SAME centroid that defines the correct
       sense everywhere else in this study (the "portability" reading —
       is the resolved sense encoded in the same region of activation space
       regardless of token position)?
     - final-position centroids (load_final_centroids), built by pooling the
       SAME profiling pass at the final position instead — a SECONDARY,
       "local recoverability" score: is sense information linearly decodable
       at the final position AT ALL, using a boundary calibrated on that
       position's own geometry, even if that geometry differs from the
       homonym position's?
     These are genuinely different questions, not two measurements of the
     same thing. target_h is only ever scored against homonym-position
     centroids (there is no "final-position" version of the homonym token).
  4. Test (dissociation): target_M_l < epsilon AND final_M_l > epsilon,
     using the PRIMARY (homonym-centroid) final score, since that is what
     the hypothesis text above ("recovers adequacy for the correct sense")
     means — "the correct sense" is defined by the same centroids used
     throughout. A parallel dissociation flag using the local-recoverability
     score is also reported for comparison.

Output
------
results/study/H4/{safe_model}/h4_{word}.csv
  columns: sentence_id, M_l_target, M_l_final, M_l_final_localcentroid,
           adequate_target, adequate_final, adequate_final_localcentroid,
           dissociation, dissociation_localcentroid

results/study/H4/h4_aggregate.csv
  dissociation rate per (model, arch_type, word), both scorings

Required data
-------------
- data/paired_sentences.json (R-condition sentences)
- results/activations/{word}/{safe_model}/ (homonym-position centroids)
- results/activations_final/{word}/{safe_model}/ (final-position centroids)
- Models: same as H3

Note on encoder behaviour
--------------------------
Scoring final_h against homonym-position centroids collapses frac_adeq_final
to exactly 0.500 in 16/16 (model, word) cells for encoders (n≈10/cell — a
coincidence this exact, this consistently, has negligible probability under
sampling noise around a genuine ~50% split). That is strong evidence the
homonym-centroid comparison has ~zero discriminative power at the final
position for encoders, but it does NOT by itself tell us whether that is
because (a) the resolved sense genuinely is not present anywhere in the
final token's representation, or (b) it is present but expressed in a
different geometric subspace than the homonym token's. The local-centroid
score is what distinguishes these: if it also shows ~chance discrimination,
(a) is favored; if it shows real separation where the homonym-centroid score
does not, (b) is favored, and target_M_l ≈ final_M_l (in the local-centroid
sense) becomes the relevant expectation for encoders instead, per H4's
architectural reasoning (homonym token already aggregates full bidirectional
context).
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.adequacy import (
    symmetric_adequacy_margins,
    symmetric_normalized_adequacy_margins,
    load_centroids,
    load_final_centroids,
)
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
                centroids       = load_centroids(results_dir, model_name, word)
                final_centroids = load_final_centroids(results_dir, model_name, word)
                layer_idx       = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as e:
                logger.warning("[H4] %s", e)
                continue

            if layer_idx not in centroids or layer_idx not in final_centroids:
                continue

            c0,  c1  = centroids[layer_idx][0],       centroids[layer_idx][1]
            fc0, fc1 = final_centroids[layer_idx][0],  final_centroids[layer_idx][1]

            target_acts, final_acts = get_dual_position_activations(
                model, tokenizer, sentences, targets,
                batch_size=4, layer_indices=[layer_idx],
            )
            H_target = target_acts[layer_idx].numpy()
            H_final  = final_acts[layer_idx].numpy()

            # R-condition sentences are balanced across both senses, so each
            # sentence must be scored against its own true sense. target_h is
            # scored against homonym-position centroids (only option — there
            # is no "final-position" version of the homonym token). final_h
            # is scored against BOTH centroid pairs: homonym-position (c0/c1,
            # PRIMARY — matches every other hypothesis's definition of
            # "adequate") and final-position (fc0/fc1, SECONDARY — local
            # recoverability diagnostic). See module docstring.
            senses_arr = np.array(senses)
            m_target = symmetric_adequacy_margins(H_target, senses_arr, c0, c1)
            m_final  = symmetric_adequacy_margins(H_final,  senses_arr, c0, c1)
            m_final_local = symmetric_adequacy_margins(H_final, senses_arr, fc0, fc1)
            # Normalized by each scoring's own inter-centroid distance —
            # comparable across architectures/layers (see adequacy.py note).
            m_target_norm = symmetric_normalized_adequacy_margins(H_target, senses_arr, c0, c1)
            m_final_norm  = symmetric_normalized_adequacy_margins(H_final,  senses_arr, c0, c1)
            m_final_local_norm = symmetric_normalized_adequacy_margins(H_final, senses_arr, fc0, fc1)

            csv_rows = []
            n_dissociation = 0
            n_dissociation_local = 0
            for sid, mt, mf, mfl, mtn, mfn, mfln in zip(
                sent_ids, m_target, m_final, m_final_local,
                m_target_norm, m_final_norm, m_final_local_norm,
            ):
                adeq_t  = bool(mt  > epsilon)
                adeq_f  = bool(mf  > epsilon)
                adeq_fl = bool(mfl > epsilon)
                dissoc       = (not adeq_t) and adeq_f   # PRIMARY: inadequate at homonym, adequate at final (homonym centroids)
                dissoc_local = (not adeq_t) and adeq_fl  # SECONDARY: same test, local-recoverability score
                if dissoc:
                    n_dissociation += 1
                if dissoc_local:
                    n_dissociation_local += 1
                csv_rows.append({
                    "sentence_id":                 sid,
                    "M_l_target":                  round(float(mt), 4),
                    "M_l_final":                   round(float(mf), 4),
                    "M_l_final_localcentroid":     round(float(mfl), 4),
                    "M_l_target_norm":             round(float(mtn), 4),
                    "M_l_final_norm":              round(float(mfn), 4),
                    "M_l_final_localcentroid_norm":round(float(mfln), 4),
                    "adequate_target":             adeq_t,
                    "adequate_final":               adeq_f,
                    "adequate_final_localcentroid": adeq_fl,
                    "dissociation":                dissoc,
                    "dissociation_localcentroid":   dissoc_local,
                    "layer":                        layer_idx,
                })

            csv_path = model_out / f"h4_{word}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

            n = len(csv_rows)
            aggregate_rows.append({
                "model":                        safe_model,
                "arch_type":                    arch_type,
                "word":                         word,
                "n_R_sentences":                n,
                "mean_M_target":                round(float(m_target.mean()), 4),
                "mean_M_final":                 round(float(m_final.mean()), 4),
                "mean_M_final_localcentroid":   round(float(m_final_local.mean()), 4),
                "mean_M_target_norm":           round(float(m_target_norm.mean()), 4),
                "mean_M_final_norm":            round(float(m_final_norm.mean()), 4),
                "mean_M_final_localcentroid_norm": round(float(m_final_local_norm.mean()), 4),
                "frac_adeq_target":             round(float((m_target > epsilon).mean()), 3),
                "frac_adeq_final":              round(float((m_final  > epsilon).mean()), 3),
                "frac_adeq_final_localcentroid":round(float((m_final_local > epsilon).mean()), 3),
                "dissociation_rate":            round(n_dissociation / n, 3),
                "dissociation_rate_localcentroid": round(n_dissociation_local / n, 3),
                "layer_used":                   layer_idx,
            })
            logger.info(
                "[H4] %s / %s | dissociation=%d/%d (local=%d/%d) | target_mean=%.3f final_mean=%.3f (local=%.3f)",
                model_name, word, n_dissociation, n, n_dissociation_local, n,
                m_target.mean(), m_final.mean(), m_final_local.mean(),
            )

    if aggregate_rows:
        agg_path = OUTPUT_BASE / "h4_aggregate.csv"
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)
        logger.info("[H4] Aggregate saved to %s", agg_path)
