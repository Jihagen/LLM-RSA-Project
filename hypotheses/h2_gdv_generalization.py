"""H2 — cross-word layer selection by GDV versus supervised adequacy.

For every held-out word, layer selection uses only the other words. The
held-out word is scored with leave-one-sentence-out centroids, so no sentence
contributes to the centroid used to classify it.

Strategies
----------
``gdv``
    Most negative mean GDV on the profiling words (label-dependent geometry).
``supervised``
    Highest mean leave-one-out adequacy on the profiling words.
``last``
    Final layer; no selection.
``oracle``
    Best layer on the held-out word itself. This is an optimistic diagnostic
    ceiling, not a deployable competitor and not evidence of generalisation.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.adequacy import adequacy_best_layer, layer_adequacy_profile

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
OUTPUT_BASE = Path("results/study/H2")


def _profile_gdv_best_layer(
    results_dir: str,
    safe_model: str,
    profiling_words: List[str],
) -> int:
    """Select the most negative mean-GDV layer on profiling words only."""
    gdv_dir = Path(results_dir) / f"{safe_model}_gdv"
    values_by_layer: Dict[int, List[float]] = {}
    for word in profiling_words:
        csv_path = gdv_dir / f"gdv_values_{word}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing corrected GDV file: {csv_path}")
        with open(csv_path) as handle:
            for row in csv.DictReader(handle):
                layer = int(row["Layer"])
                if layer != 0:
                    values_by_layer.setdefault(layer, []).append(float(row["GDV"]))
    complete = {
        layer: values
        for layer, values in values_by_layer.items()
        if len(values) == len(profiling_words)
    }
    if not complete:
        raise ValueError(f"No common GDV layers across {profiling_words}")
    return min(complete, key=lambda layer: np.mean(complete[layer]))


def _profile_supervised_best_layer(
    profiles_by_word: Dict[str, Dict[int, Dict]],
    profiling_words: List[str],
) -> int:
    """Select by mean held-out-sentence adequacy on profiling words only."""
    common_layers = set.intersection(
        *(set(profiles_by_word[word]) for word in profiling_words)
    )
    candidates = sorted(layer for layer in common_layers if layer != 0)
    if not candidates:
        raise ValueError(f"No common contextual layers across {profiling_words}")

    def selection_score(layer: int):
        fractions = [
            profiles_by_word[word][layer]["fraction_adequate"]
            for word in profiling_words
        ]
        normalized = [
            profiles_by_word[word][layer]["mean_norm"]
            for word in profiling_words
        ]
        return float(np.mean(fractions)), float(np.mean(normalized))

    return max(candidates, key=selection_score)


def _strategy_metrics(profile: Dict[int, Dict], layer: int) -> Dict[str, float]:
    if layer not in profile:
        return {
            "mean_M_raw": float("nan"),
            "mean_M_norm": float("nan"),
            "fraction_adequate": float("nan"),
        }
    result = profile[layer]
    return {
        "mean_M_raw": float(result["mean_raw"]),
        "mean_M_norm": float(result["mean_norm"]),
        "fraction_adequate": float(result["fraction_adequate"]),
    }


def run_h2(
    model_names: List[str],
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    epsilon: float = 0.0,
) -> None:
    """Run leave-one-word-out strategy comparison for every model."""
    words = words or ["bank", "bark", "bat", "crane", "spring", "match", "light", "pitch"]
    if len(words) < 2:
        raise ValueError("H2 requires at least two words")

    output_base = Path(results_dir) / "study" / "H2"
    output_base.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out = output_base / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        profiles_by_word = {}
        for word in words:
            try:
                profiles_by_word[word] = layer_adequacy_profile(
                    results_dir,
                    model_name,
                    word,
                    epsilon=epsilon,
                    centroid_mode="leave_one_out",
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("[H2] %s / %s skipped: %s", model_name, word, exc)

        available_words = [word for word in words if profiles_by_word.get(word)]
        if len(available_words) < 2:
            logger.warning("[H2] %s has fewer than two usable words", model_name)
            continue

        rows = []
        for held_out in available_words:
            profiling = [word for word in available_words if word != held_out]
            held_out_profile = profiles_by_word[held_out]
            try:
                gdv_layer = _profile_gdv_best_layer(results_dir, safe_model, profiling)
                supervised_layer = _profile_supervised_best_layer(
                    profiles_by_word, profiling
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("[H2] %s held-out=%s skipped: %s", model_name, held_out, exc)
                continue

            oracle_layer = adequacy_best_layer(held_out_profile)
            last_layer = max(held_out_profile)
            layers = {
                "gdv": gdv_layer,
                "supervised": supervised_layer,
                "last": last_layer,
                "oracle": oracle_layer,
            }
            row = {
                "held_out_word": held_out,
                "centroid_mode": "leave_one_out",
                **{f"{name}_selected_layer": layer for name, layer in layers.items()},
            }
            for name, layer in layers.items():
                metrics = _strategy_metrics(held_out_profile, layer)
                row[f"mean_M_raw_{name}"] = round(metrics["mean_M_raw"], 6)
                row[f"mean_M_norm_{name}"] = round(metrics["mean_M_norm"], 6)
                row[f"fraction_adequate_{name}"] = round(
                    metrics["fraction_adequate"], 4
                )
                # Exact agreement alone is brittle when adjacent layers behave
                # almost identically.  Report distance, held-out performance
                # regret, and held-out adequacy rank as complementary outcomes.
                row[f"layer_distance_to_oracle_{name}"] = abs(layer - oracle_layer)
                row[f"relative_depth_{name}"] = round(layer / max(last_layer, 1), 4)

            oracle_fraction = row["fraction_adequate_oracle"]
            ranking = sorted(
                [layer for layer in held_out_profile if layer != 0],
                key=lambda candidate: (
                    held_out_profile[candidate]["fraction_adequate"],
                    held_out_profile[candidate]["mean_norm"],
                ),
                reverse=True,
            )
            rank_by_layer = {layer: rank + 1 for rank, layer in enumerate(ranking)}
            for name, layer in layers.items():
                row[f"fraction_regret_to_oracle_{name}"] = round(
                    oracle_fraction - row[f"fraction_adequate_{name}"], 4
                )
                row[f"held_out_adequacy_rank_{name}"] = rank_by_layer[layer]
                row[f"within_2_layers_of_oracle_{name}"] = abs(layer - oracle_layer) <= 2
            rows.append(row)

        if not rows:
            continue
        out_path = model_out / "h2_loo.csv"
        with open(out_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        summary = {"model": safe_model, "n_held_out_words": len(rows)}
        for strategy in ("gdv", "supervised", "last", "oracle"):
            for metric in ("mean_M_norm", "fraction_adequate"):
                values = [row[f"{metric}_{strategy}"] for row in rows]
                summary[f"{metric}_{strategy}"] = round(float(np.mean(values)), 6)
            for metric in (
                "layer_distance_to_oracle",
                "fraction_regret_to_oracle",
                "held_out_adequacy_rank",
            ):
                values = [row[f"{metric}_{strategy}"] for row in rows]
                summary[f"mean_{metric}_{strategy}"] = round(float(np.mean(values)), 6)
            summary[f"fraction_within_2_layers_of_oracle_{strategy}"] = round(
                float(np.mean([row[f"within_2_layers_of_oracle_{strategy}"] for row in rows])),
                6,
            )
        summary["gdv_minus_last_fraction"] = round(
            summary["fraction_adequate_gdv"] - summary["fraction_adequate_last"], 6
        )
        summary["supervised_minus_last_fraction"] = round(
            summary["fraction_adequate_supervised"]
            - summary["fraction_adequate_last"],
            6,
        )
        summary_rows.append(summary)
        logger.info("[H2] Corrected LOO results saved to %s", out_path)

    if summary_rows:
        summary_path = output_base / "h2_strategy_summary.csv"
        with open(summary_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("[H2] Strategy summary saved to %s", summary_path)
