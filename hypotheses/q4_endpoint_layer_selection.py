"""Q4 -- endpoint (sentence-final) layer selection, as a fair comparison
point for causal decoders.

Motivation
----------
H3 shows that causal decoders cannot access right-hand disambiguating
context at the homonym position: R-condition decoder adequacy there is
exactly 0.500 by construction (the two R sentences share an identical
causally-masked prefix up to the homonym). H1's layer selection is
nevertheless computed at that same homonym position for every architecture.
For decoders this means H1 is choosing "the best layer to represent a
position where the resolving information is not even present yet" -- which
could make decoders look like they lack a usable semantic layer, when the
real disambiguation for a decoder only becomes possible at a later position.

Q4 asks the same question H1 asks -- which layer best supports held-out
sense decoding? -- but scored at the sentence-final period position, using
H4's existing R-condition sentences (the disambiguating clause has already
been read by that point) and the existing period-position profiling
centroids, at *every* available layer rather than only the single
H1-selected layer.

Two stages
----------
1. ``extract_endpoint_activations`` -- one new forward pass per model over
   the R-condition sentences (10 per word), with ``layer_indices=None`` so
   every layer's period-position state is captured in one pass, cached to
   ``results/activations_paired_final/{word}/{model}/layer_{i}.h5``. This is
   the only step that needs a GPU/model load; a word/model already cached is
   skipped, mirroring run_h2.py's cache check.
2. ``run_q4`` -- pure CPU. For every model/word/layer, scores the cached
   endpoint states against the *existing* period-position profiling
   centroids (``results/activations_final/``, already computed by
   ``run_h2.py``) using the same ``symmetric_adequacy_margins`` H4 uses,
   aggregates fraction_adequate and mean normalised margin, and selects the
   best layer with ``experiments.adequacy.adequacy_best_layer`` -- the exact
   same tie-break H1 uses (fraction_adequate, then mean_norm, then earliest
   layer among remaining ties), so the two analyses are comparable by
   construction rather than by a separately-reimplemented rule.

No new stimuli and no new metrics: same R-condition sentences H3/H4 already
use, same margin/adequacy functions, same selection rule as H1.

Output
------
results/study/Q4/q4_layer_curves.csv
    One row per (model, word, layer): model, architecture, homonym, layer,
    relative_depth, endpoint_adequacy, endpoint_mean_margin,
    selected_endpoint_layer (the last column is the same value repeated
    across every layer row of that model/word group, for convenience).

results/study/Q4/{safe_model}/q4_selected_layer.csv
    One row per word: the layer actually selected. Same shape as
    h1_summary.csv's best_layer_M column, for use as an alternate
    layer_lookup callable when rerunning H5 (see run_h5's layer_lookup
    parameter).
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from experiments.adequacy import (
    adequacy_best_layer,
    load_final_centroids,
    symmetric_adequacy_margins,
    symmetric_normalized_adequacy_margins,
)
from hypotheses.h3_context_position import H3_MODELS, PAIRED_DATA_PATH
from models import get_dual_position_activations, is_decoder_only, load_model_and_tokenizer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
OUTPUT_BASE = Path("results/study/Q4")
ENDPOINT_CACHE_SUBDIR = "activations_paired_final"
DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]


def _r_condition_items(paired_data_path: str, word: str) -> List[Dict]:
    with open(paired_data_path) as f:
        data = json.load(f)
    if word not in data:
        raise KeyError(f"Word '{word}' not found in {paired_data_path}")
    return [item for item in data[word] if item["condition"] == "R"]


def _endpoint_cache_dir(results_dir: str, word: str, safe_model: str) -> Path:
    return Path(results_dir) / ENDPOINT_CACHE_SUBDIR / word / safe_model


def _has_endpoint_cache(results_dir: str, word: str, safe_model: str) -> bool:
    cache_dir = _endpoint_cache_dir(results_dir, word, safe_model)
    return cache_dir.exists() and any(cache_dir.glob("layer_[1-9]*.h5"))


def _save_endpoint_activations(
    results_dir: str,
    word: str,
    safe_model: str,
    activations: Dict[int, "torch.Tensor"],
    labels: np.ndarray,
    sentences: List[str],
) -> None:
    """Cache every layer's period-position R-condition states, in the same
    on-disk layout save_target_activations uses, so downstream code (or a
    human) can load them the same way as any other activation cache."""
    target_dir = _endpoint_cache_dir(results_dir, word, safe_model)
    target_dir.mkdir(parents=True, exist_ok=True)
    dt = h5py.string_dtype(encoding="utf-8")
    for layer_idx, tensor in activations.items():
        arr = tensor.cpu().numpy()
        with h5py.File(target_dir / f"layer_{layer_idx}.h5", "w") as f:
            f.create_dataset("X", data=arr)
            f.create_dataset("labels", data=labels)
            f.create_dataset(
                "sentences", data=np.array([s.encode("utf-8") for s in sentences]), dtype=dt
            )


def _load_endpoint_layer(results_dir: str, word: str, safe_model: str, layer: int):
    path = _endpoint_cache_dir(results_dir, word, safe_model) / f"layer_{layer}.h5"
    with h5py.File(path, "r") as f:
        return f["X"][:], f["labels"][:]


def extract_endpoint_activations(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    paired_data_path: str = PAIRED_DATA_PATH,
    force: bool = False,
) -> None:
    """GPU step: cache every layer's sentence-final state for the R-condition
    sentences. Requires loading each model checkpoint; skipped per
    model/word if already cached, unless force=True."""
    model_names = model_names or H3_MODELS
    words = words or DEFAULT_WORDS

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        words_todo = [
            w for w in words
            if force or not _has_endpoint_cache(results_dir, w, safe_model)
        ]
        if not words_todo:
            logger.info("[Q4] %s: all words already cached, skipping model load.", model_name)
            continue

        model, tokenizer = load_model_and_tokenizer(model_name)
        logger.info("[Q4] %s: extracting endpoint activations for %s", model_name, words_todo)
        for word in words_todo:
            try:
                items = _r_condition_items(paired_data_path, word)
            except KeyError as exc:
                logger.warning("[Q4] %s", exc)
                continue
            if not items:
                logger.warning("[Q4] No R-condition sentences for '%s'.", word)
                continue

            sentences = [item["sentence"] for item in items]
            senses = np.asarray([item["sense"] for item in items])

            # layer_indices=None -> every layer in one forward pass; we only
            # need the sentence-final ("final_acts") position here.
            _target_acts, final_acts = get_dual_position_activations(
                model, tokenizer, sentences, [word] * len(sentences),
                batch_size=4, layer_indices=None,
            )
            _save_endpoint_activations(
                results_dir, word, safe_model, final_acts, senses, sentences
            )
            logger.info(
                "[Q4] %s/%s: cached %d layers x %d R-condition sentences",
                model_name, word, len(final_acts), len(sentences),
            )
        del model, tokenizer


def _score_layer(
    results_dir: str, word: str, safe_model: str, layer: int, fc0: np.ndarray, fc1: np.ndarray,
    epsilon: float = 0.0,
):
    X, labels = _load_endpoint_layer(results_dir, word, safe_model, layer)
    margins_norm = symmetric_normalized_adequacy_margins(X, labels, fc0, fc1)
    margins_raw = symmetric_adequacy_margins(X, labels, fc0, fc1)
    return {
        "fraction_adequate": float((margins_raw > epsilon).mean()),
        "mean_norm": float(margins_norm.mean()),
    }


def run_q4(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    epsilon: float = 0.0,
) -> None:
    """CPU step: score every cached endpoint layer against the existing
    period-position profiling centroids, and select the best layer with
    the same rule H1 uses (adequacy_best_layer)."""
    model_names = model_names or H3_MODELS
    words = words or DEFAULT_WORDS
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    curve_rows: List[Dict] = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        arch_type = "decoder" if _is_decoder_name(model_name) else "encoder"
        selected_rows: List[Dict] = []

        for word in words:
            if not _has_endpoint_cache(results_dir, word, safe_model):
                logger.warning(
                    "[Q4] No endpoint cache for %s/%s; run extract_endpoint_activations first.",
                    model_name, word,
                )
                continue
            try:
                final_centroids = load_final_centroids(results_dir, model_name, word)
            except FileNotFoundError as exc:
                logger.warning("[Q4] %s", exc)
                continue

            cache_dir = _endpoint_cache_dir(results_dir, word, safe_model)
            layers = sorted(
                int(p.stem.removeprefix("layer_")) for p in cache_dir.glob("layer_*.h5")
            )
            last_layer = layers[-1]

            profile: Dict[int, Dict] = {}
            for layer in layers:
                if layer not in final_centroids:
                    continue
                fc0, fc1 = final_centroids[layer][0], final_centroids[layer][1]
                profile[layer] = _score_layer(
                    results_dir, word, safe_model, layer, fc0, fc1, epsilon
                )
            if not profile:
                logger.warning("[Q4] Empty profile for %s/%s", model_name, word)
                continue

            selected_layer = adequacy_best_layer(profile)

            for layer in layers:
                if layer not in profile:
                    continue
                curve_rows.append({
                    "model": safe_model,
                    "architecture": arch_type,
                    "homonym": word,
                    "layer": layer,
                    "relative_depth": round(layer / last_layer, 6),
                    "endpoint_adequacy": round(profile[layer]["fraction_adequate"], 6),
                    "endpoint_mean_margin": round(profile[layer]["mean_norm"], 6),
                    "selected_endpoint_layer": selected_layer,
                })
            selected_rows.append({
                "word": word,
                "best_layer_M": selected_layer,
                "relative_depth": round(selected_layer / last_layer, 6),
                "endpoint_adequacy": round(profile[selected_layer]["fraction_adequate"], 6),
                "endpoint_mean_margin": round(profile[selected_layer]["mean_norm"], 6),
            })
            logger.info(
                "[Q4] %s/%s: selected endpoint layer %d/%d (adequacy=%.3f)",
                model_name, word, selected_layer, last_layer,
                profile[selected_layer]["fraction_adequate"],
            )

        if selected_rows:
            model_out = OUTPUT_BASE / safe_model
            model_out.mkdir(parents=True, exist_ok=True)
            with open(model_out / "q4_selected_layer.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=selected_rows[0].keys())
                writer.writeheader()
                writer.writerows(selected_rows)

    if curve_rows:
        with open(OUTPUT_BASE / "q4_layer_curves.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=curve_rows[0].keys())
            writer.writeheader()
            writer.writerows(curve_rows)
        logger.info("[Q4] Wrote %d rows to %s", len(curve_rows), OUTPUT_BASE / "q4_layer_curves.csv")


def _is_decoder_name(model_name: str) -> bool:
    """Architecture lookup by name only, no model load required (mirrors
    the encoder/decoder split used throughout the visualisation scripts)."""
    encoders = {
        "answerdotai/ModernBERT-large",
        "microsoft/deberta-v3-large",
        "FacebookAI/roberta-large",
        "FacebookAI/xlm-roberta-large",
    }
    return model_name not in encoders


def q4_selected_layer(model_name: str, results_dir: str, word: str) -> int:
    """layer_lookup-compatible callable (same signature as
    hypotheses.h3_context_position._select_layer) reading Q4's selected
    layer instead of H1's. Pass this to run_h5(layer_lookup=...) to rerun
    H5 on the endpoint-selected layer instead of the homonym-selected one."""
    safe = model_name.replace("/", "_")
    path = Path(results_dir) / "study" / "Q4" / safe / "q4_selected_layer.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No Q4 selected-layer summary at {path}; run run_q4(...) first."
        )
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["word"] == word:
                return int(row["best_layer_M"])
    raise FileNotFoundError(f"Word '{word}' not found in {path}")
