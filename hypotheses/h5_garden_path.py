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
  1. Use get_homonym_and_resolution_activations to extract, in one forward
     pass, the homonym-position representation and the representation at the
     annotated resolution_word for that sentence — the specific word that
     actually carries the disambiguating information (e.g. "river" for a
     "bank" sentence), located by word-boundary string search wherever it
     falls in the sentence.
  2. Score both positions against the same homonym-position sense centroids
     (the same M_l definition used everywhere else in the study). Both
     margins are also reported normalized by their own inter-centroid
     distance.
  3. Report: does the representation shift from the primed (wrong) sense at
     the homonym to the resolved (correct) sense at the resolution word?

For encoders: shift should be visible across layers (early layers commit to
primed sense; late layers correct to resolved sense).
For decoders: homonym-token is stuck with primed sense; the resolution word,
which comes later, may shift.

Why the resolution word, not the sentence-final token
------------------------------------------------------
An earlier version of this experiment scored the sentence's last non-special
token, on the assumption that "final position" was a reasonable proxy for
"has seen the disambiguating clause." That assumption fails whenever the
sentence happens to end on a word that isn't the disambiguator itself — a
neutral filler ("...paddle into the sunset together") or, worse, a word
associated with the *primed* sense ("...flew over the scaffolding" ends a
bird-revealing sentence on a construction noun). Scoring at the annotated
resolution word instead removes the dependency on how each sentence happens
to end, and removes the need for a separate "local final-centroid" baseline
that the earlier design used to guard against exactly that mismatch: since
the resolution word is a real content word, it can be scored directly
against the same homonym-position centroids used everywhere else.

Three-way resolution bucketing — reported at multiple thresholds, not one
-----------------------------------------------------------------------
The binary priming_conflict flag (target inadequate AND resolution adequate)
collapses two very different outcomes into one "not conflict" bucket:
a resolution-word representation that sits near the decision boundary
(genuinely undecided) and one that sits confidently on the WRONG side
(decisively stuck on the primed sense, i.e. failed revision, not
indecision). These are mechanistically different and worth separating.

Each sentence's resolution-word normalized margin is bucketed into:
  - "confident_correct" : margin >  threshold
  - "confident_primed"  : margin < -threshold
  - "ambiguous"          : |margin| <= threshold (near the boundary — not
                           confidently either sense)

An earlier version fixed this at a single threshold (0.3) and treated the
resulting "ambiguous" fraction as the headline number. That threshold was
carried over from contexts (e.g. H0's carrier-bias cutoff) where it was
implicitly calibrated against aggregated, strongly-disambiguated signals;
checked against the actual per-sentence resolution-margin distribution here
(median ~0.06-0.10, roughly a third of H3's L-condition cell-level mean of
0.318), 0.3 turned out to be a demanding bar for a single-word,
single-sentence measurement, producing >85% "ambiguous" largely as an
artifact of the cutoff rather than genuine indecision. RESOLUTION_THRESHOLDS
below now reports bucket fractions at BOTH a lower (0.1) and the original
(0.3) threshold side by side, so no single cutoff is silently treated as
canonical — and the raw per-sentence normalized margin (M_l_resolution_correct_norm)
is always available unaggregated in the per-word CSV for anyone who wants a
different cutoff, or none at all. `priming_conflict_rate` (sign-only, ε = 0)
remains the most robust single summary number, since it makes no threshold
assumption at all.
RESOLUTION_THRESHOLDS = (0.1, 0.3)

Output
------
results/study/H5/{safe_model}/h5_{word}.csv
  columns: sentence_id, correct_sense, primed_sense, resolution_word,
           M_l_target_correct, M_l_resolution_correct, *_norm variants
           (the unaggregated per-sentence margins), priming_conflict, and
           one resolution_bucket_thr{T} column per threshold in
           RESOLUTION_THRESHOLDS.

results/study/H5/h5_aggregate.csv
  includes std_M_resolution_norm per cell (to help separate a genuine
  near-zero/ambiguous signature from a noisy, high-variance,
  too-little-data signature) and, per threshold in RESOLUTION_THRESHOLDS,
  the fraction of sentences in each of the three resolution buckets
  (frac_confident_correct_thr{T}, frac_confident_primed_thr{T},
  frac_ambiguous_thr{T}).

Required data
-------------
- data/garden_path_sentences.json (6-7 sentences/word), each item annotated
  with a resolution_word field.
- results/activations/{word}/{safe_model}/ (homonym-position centroids)

Note
----
This experiment is EXPLORATORY. Garden-path sentences have three
per-sentence annotations:
  - primed_sense:    the sense implied by left context
  - correct_sense:   the sense intended by the full sentence
  - resolution_word: the specific word that carries the disambiguation
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.adequacy import load_centroids, normalized_adequacy_margin
from hypotheses.h3_context_position import H3_MODELS, _select_layer
from models import get_homonym_and_resolution_activations, is_decoder_only, load_model_and_tokenizer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR      = "results"
GP_DATA_PATH     = "data/garden_path_sentences.json"
OUTPUT_BASE      = Path("results/study/H5")

# See module docstring, "Three-way resolution bucketing — reported at
# multiple thresholds, not one". No single value here is "the" threshold;
# both are reported side by side in the per-sentence and aggregate output.
RESOLUTION_THRESHOLDS = (0.1, 0.3)


def _resolution_bucket(margin_norm: float, threshold: float) -> str:
    if margin_norm > threshold:
        return "confident_correct"
    if margin_norm < -threshold:
        return "confident_primed"
    return "ambiguous"


def _threshold_suffix(threshold: float) -> str:
    return f"thr{threshold:g}"


def _load_garden_path(path: str, word: str) -> List[Dict]:
    """
    Load garden-path sentences for a word.

    Expected JSON structure:
    {
      "bank": [
        {
          "id": "bank_gp_01",
          "sentence": "...",
          "primed_sense": 1,       // sense implied by left context
          "correct_sense": 0,      // sense intended by full sentence
          "resolution_word": "river"  // word that carries the disambiguation
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

            sentences    = [item["sentence"]         for item in items]
            sent_ids     = [item.get("id", f"{word}_gp_{i}") for i, item in enumerate(items)]
            correct_ids  = [item["correct_sense"]     for item in items]
            primed_ids   = [item["primed_sense"]      for item in items]
            resolvers    = [item["resolution_word"]   for item in items]
            targets      = [word] * len(sentences)

            try:
                centroids = load_centroids(results_dir, model_name, word)
                layer_idx = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as e:
                logger.warning("[H5] %s", e)
                continue

            if layer_idx not in centroids:
                continue

            target_acts, resolution_acts = get_homonym_and_resolution_activations(
                model, tokenizer, sentences, targets, resolvers,
                batch_size=4, layer_indices=[layer_idx],
            )
            H_target     = target_acts[layer_idx].numpy()
            H_resolution = resolution_acts[layer_idx].numpy()

            csv_rows = []
            n_conflict = 0
            for i, (sid, c_sense, p_sense, rword) in enumerate(zip(sent_ids, correct_ids, primed_ids, resolvers)):
                if c_sense not in centroids[layer_idx] or p_sense not in centroids[layer_idx]:
                    continue

                c_correct = centroids[layer_idx][c_sense]
                c_primed  = centroids[layer_idx][p_sense]

                mt_correct = adequacy_margin_single(H_target[i],     c_correct, c_primed)
                mr_correct = adequacy_margin_single(H_resolution[i], c_correct, c_primed)

                mt_correct_norm = normalized_adequacy_margin(H_target[i],     c_correct, c_primed)
                mr_correct_norm = normalized_adequacy_margin(H_resolution[i], c_correct, c_primed)

                # Priming conflict: homonym aligns with primed sense, resolution word aligns with correct
                conflict = (mt_correct < epsilon) and (mr_correct > epsilon)
                if conflict:
                    n_conflict += 1

                row = {
                    "sentence_id":               sid,
                    "correct_sense":             c_sense,
                    "primed_sense":              p_sense,
                    "resolution_word":           rword,
                    "M_l_target_correct":        round(float(mt_correct), 4),
                    "M_l_resolution_correct":    round(float(mr_correct), 4),
                    "M_l_target_correct_norm":     round(float(mt_correct_norm), 4),
                    "M_l_resolution_correct_norm": round(float(mr_correct_norm), 4),
                    "priming_conflict":          conflict,
                }
                for t in RESOLUTION_THRESHOLDS:
                    row[f"resolution_bucket_{_threshold_suffix(t)}"] = _resolution_bucket(mr_correct_norm, t)
                row["layer"] = layer_idx
                csv_rows.append(row)

            if not csv_rows:
                continue

            csv_path = model_out / f"h5_{word}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

            n = len(csv_rows)
            resolution_norms = [r["M_l_resolution_correct_norm"] for r in csv_rows]
            agg_row = {
                "model":              safe_model,
                "arch_type":          arch_type,
                "word":               word,
                "n_gp_sentences":     n,
                "priming_conflict_rate": round(n_conflict / n, 3),
                "mean_M_target":         round(float(np.mean([r["M_l_target_correct"]     for r in csv_rows])), 4),
                "mean_M_resolution":     round(float(np.mean([r["M_l_resolution_correct"] for r in csv_rows])), 4),
                "mean_M_resolution_norm": round(float(np.mean(resolution_norms)), 4),
                # Spread of the per-sentence normalized margins — low std + mean near
                # zero suggests genuine unresolved ambiguity; high std suggests a
                # noisy/underpowered signal rather than a consistent one either way.
                "std_M_resolution_norm": round(float(np.std(resolution_norms)), 4),
            }
            # Three-way resolution bucket fractions, per threshold (see module
            # docstring) — no single threshold is treated as canonical.
            for t in RESOLUTION_THRESHOLDS:
                suffix  = _threshold_suffix(t)
                buckets = [r[f"resolution_bucket_{suffix}"] for r in csv_rows]
                agg_row[f"frac_confident_correct_{suffix}"] = round(buckets.count("confident_correct") / n, 3)
                agg_row[f"frac_confident_primed_{suffix}"]  = round(buckets.count("confident_primed") / n, 3)
                agg_row[f"frac_ambiguous_{suffix}"]         = round(buckets.count("ambiguous") / n, 3)
            agg_row["layer_used"] = layer_idx
            aggregate_rows.append(agg_row)
            logger.info(
                "[H5] %s / %s | conflict_rate=%d/%d",
                model_name, word, n_conflict, n,
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
