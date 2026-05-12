"""
Master runner for H1–H5.

Usage examples
--------------
  python run_study.py                          # run H1+H2 (no forward passes needed)
  python run_study.py --hypotheses H1 H2 H3   # specific subset
  python run_study.py --hypotheses H3 H4 H5   # forward-pass experiments only
  python run_study.py --models deberta roberta # restrict to named model subsets

Per-hypothesis data requirements
---------------------------------
H1  Layer adequacy profile
    Data   : results/activations/{word}/{model}/  (H5 files, all 8 models × 4+ words)
    Models : all 8 (encoder + decoder)
    Status : READY — bank/bark/bat/crane fully cached

H2  GDV generalisation (leave-one-out)
    Data   : H5 files + gdv_values_{word}.csv for all words
    Models : all 8
    Status : READY with 4 words; add remaining 4 (re-run run_h2.py) for full coverage

H3  Right-context vulnerability
    Data   : data/paired_sentences.json  +  H5 profiling files
    Models : DeBERTa, RoBERTa-large, Mistral-Nemo, Qwen-3B
    Status : BLOCKED — create paired_sentences.json first
             (see data/inspect_paired_sentences.ipynb)

H4  Token-position dissociation
    Data   : data/paired_sentences.json (R-condition only)  +  H5 files
    Models : same 4 as H3
    Status : BLOCKED — depends on H3 data

H5  Garden-path / representational revision (exploratory)
    Data   : data/garden_path_sentences.json  +  H5 files
    Models : same 4 as H3/H4
    Status : BLOCKED — create garden_path_sentences.json first
"""

import argparse
import gc
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_study")

# ── Model sets ────────────────────────────────────────────────────────────────

ALL_MODELS = [
    "answerdotai/ModernBERT-large",
    "microsoft/deberta-v3-large",
    "FacebookAI/roberta-large",
    "FacebookAI/xlm-roberta-large",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "mistralai/Mistral-Nemo-Base-2407",
    "allenai/OLMo-2-1124-7B",
]

# Reduced set for H3-H5 (representative encoder + decoder per size tier)
H3_MODELS = [
    "microsoft/deberta-v3-large",
    "FacebookAI/roberta-large",
    "mistralai/Mistral-Nemo-Base-2407",
    "Qwen/Qwen2.5-3B",
]

MODEL_ALIASES = {
    "deberta":  "microsoft/deberta-v3-large",
    "roberta":  "FacebookAI/roberta-large",
    "xlm":      "FacebookAI/xlm-roberta-large",
    "modernbert": "answerdotai/ModernBERT-large",
    "qwen3b":   "Qwen/Qwen2.5-3B",
    "qwen7b":   "Qwen/Qwen2.5-7B",
    "mistral":  "mistralai/Mistral-Nemo-Base-2407",
    "olmo":     "allenai/OLMo-2-1124-7B",
}

WORDS_DEFAULT = ["bank", "bark", "bat", "crane"]


def _resolve_models(names):
    resolved = []
    for n in names:
        if n in MODEL_ALIASES:
            resolved.append(MODEL_ALIASES[n])
        elif n in ALL_MODELS:
            resolved.append(n)
        else:
            logger.warning("Unknown model alias '%s' — skipping.", n)
    return resolved or ALL_MODELS


def _run_h1(args):
    from hypothesis.h1_layer_adequacy import run_h1
    models = _resolve_models(args.models) if args.models else ALL_MODELS
    logger.info("=== H1: Layer Adequacy | %d models | words=%s ===", len(models), args.words)
    run_h1(model_names=models, words=args.words)


def _run_h2(args):
    from hypothesis.h2_gdv_generalization import run_h2
    models = _resolve_models(args.models) if args.models else ALL_MODELS
    logger.info("=== H2: GDV Generalisation | %d models ===", len(models))
    run_h2(model_names=models, words=args.words)


def _run_h3(args):
    from hypothesis.h3_context_position import run_h3
    models = _resolve_models(args.models) if args.models else H3_MODELS
    logger.info("=== H3: Context Position | %d models ===", len(models))
    run_h3(model_names=models, words=args.words)
    gc.collect()


def _run_h4(args):
    from hypothesis.h4_dissociation import run_h4
    models = _resolve_models(args.models) if args.models else H3_MODELS
    logger.info("=== H4: Token Dissociation | %d models ===", len(models))
    run_h4(model_names=models, words=args.words)
    gc.collect()


def _run_h5(args):
    from hypothesis.h5_garden_path import run_h5
    models = _resolve_models(args.models) if args.models else H3_MODELS
    logger.info("=== H5: Garden-Path (exploratory) | %d models ===", len(models))
    run_h5(model_names=models, words=args.words)
    gc.collect()


RUNNERS = {
    "H1": _run_h1,
    "H2": _run_h2,
    "H3": _run_h3,
    "H4": _run_h4,
    "H5": _run_h5,
}


def main():
    parser = argparse.ArgumentParser(description="Run H1–H5 study experiments.")
    parser.add_argument(
        "--hypotheses", nargs="+",
        choices=["H1", "H2", "H3", "H4", "H5"],
        default=["H1", "H2"],
        help="Which hypotheses to run (default: H1 H2).",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=(
            "Model names or aliases to run. "
            "Aliases: deberta, roberta, xlm, modernbert, qwen3b, qwen7b, mistral, olmo. "
            "Default: all 8 for H1/H2, 4-model subset for H3/H4/H5."
        ),
    )
    parser.add_argument(
        "--words", nargs="+",
        default=WORDS_DEFAULT,
        help=f"Homonyms to include (default: {WORDS_DEFAULT}).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.0,
        help="Adequacy threshold epsilon (default: 0.0).",
    )
    args = parser.parse_args()

    logger.info("Study run: hypotheses=%s, words=%s", args.hypotheses, args.words)

    for hyp in args.hypotheses:
        try:
            RUNNERS[hyp](args)
        except Exception as exc:
            logger.error("=== %s FAILED: %s ===", hyp, exc, exc_info=True)

    logger.info("Study complete.")


if __name__ == "__main__":
    main()
