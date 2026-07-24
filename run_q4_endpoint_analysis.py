"""Root-level runner for the Q4 endpoint (sentence-final) layer reanalysis.

Three stages, run in order:

1. Extract every layer's sentence-final state for H3/H4's R-condition
   sentences (GPU: loads each model checkpoint; skipped per model/word if
   already cached under results/activations_paired_final/).
2. Score every cached layer against the existing period-position profiling
   centroids and select the best layer with H1's exact tie-break rule
   (CPU-only; writes results/study/Q4/).
3. Rerun H5 using the Q4-selected layer instead of H1's homonym-position
   layer, writing to a separate output directory (results/study/H5_q4/ by
   default) so the original H1-based H5 results are never touched.

Mirrors run_h2.py's standalone-script convention; the H0-H5 dispatch in
run_study.py is intentionally left untouched, since Q4 is a new reanalysis
rather than one of the six original hypotheses.
"""

import argparse
import gc
import logging
from pathlib import Path

from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", nargs="*", default=DEFAULT_WORDS)
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Restrict to specific models (safe or slash-form names); default all 8.",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip stage 1 (assume results/activations_paired_final/ is already complete).",
    )
    parser.add_argument(
        "--skip-h5-rerun", action="store_true",
        help="Skip stage 3 (only produce the Q4 layer curves/selection, no H5 rerun).",
    )
    parser.add_argument("--force-extraction", action="store_true")
    parser.add_argument(
        "--h5-output-base", default="results/study/H5_q4",
        help="Where the Q4-layer H5 rerun is written (never the original results/study/H5/).",
    )
    parser.add_argument("--allow-incomplete-h5-design", action="store_true")
    args = parser.parse_args()

    from utils.model_registry import ALL_MODELS, MODEL_ALIASES

    if args.models:
        models = [MODEL_ALIASES.get(m, m) for m in args.models]
    else:
        models = ALL_MODELS

    if not args.skip_extraction:
        from hypotheses.q4_endpoint_layer_selection import extract_endpoint_activations
        logger.info("=== Q4 stage 1: extracting endpoint activations | %d models ===", len(models))
        extract_endpoint_activations(
            model_names=models, words=args.words, force=args.force_extraction
        )
        gc.collect()
    else:
        logger.info("=== Q4 stage 1 skipped (--skip-extraction) ===")

    from hypotheses.q4_endpoint_layer_selection import run_q4
    logger.info("=== Q4 stage 2: scoring and layer selection ===")
    run_q4(model_names=models, words=args.words)

    if not args.skip_h5_rerun:
        from hypotheses.h5_garden_path import run_h5
        from hypotheses.q4_endpoint_layer_selection import q4_selected_layer
        logger.info(
            "=== Q4 stage 3: rerunning H5 with the endpoint-selected layer -> %s ===",
            args.h5_output_base,
        )
        run_h5(
            model_names=models,
            words=args.words,
            output_base=Path(args.h5_output_base),
            layer_lookup=q4_selected_layer,
            allow_incomplete_design=args.allow_incomplete_h5_design,
        )
    else:
        logger.info("=== Q4 stage 3 skipped (--skip-h5-rerun) ===")

    logger.info("Q4 endpoint reanalysis complete.")


if __name__ == "__main__":
    main()
