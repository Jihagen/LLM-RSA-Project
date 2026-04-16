import csv
import json
import os
from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from data import flatten_dataframe
from models import collect_target_span_representations, load_model_and_tokenizer
from probing import ProbingClassifier
from utils.profile_metrics import (
    compute_gdv,
    compute_profile_sharpness,
    compute_semantic_band,
    interpolate_profile,
    linear_cka,
    maybe_global_layer,
    relative_peak_position,
    safe_pearsonr,
    top_k_layers,
)


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def save_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_word_profiles(word: str, layer_rows: Sequence[Dict[str, object]], output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    layers = [int(row["layer"]) for row in layer_rows]
    probe_scores = [float(row["probe_f1_mean"]) for row in layer_rows]
    sentence_scores = [float(row["sentence_probe_f1_mean"]) for row in layer_rows]
    gdv_scores = [float(row["gdv_separation"]) for row in layer_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, probe_scores, marker="o", label="Target-span probe F1")
    plt.plot(layers, sentence_scores, marker="o", label="Sentence-mean baseline F1")
    plt.plot(layers, gdv_scores, marker="o", label="-GDV separation")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title(f"Distributed semantic profile for '{word}'")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{word}_profiles.png"))
    plt.close()


def plot_peak_histogram(model_name: str, peak_layers: Sequence[int], metric_name: str, output_dir: str) -> None:
    if not peak_layers:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    counts = Counter(int(layer) for layer in peak_layers)
    layers = sorted(counts.keys())
    values = [counts[layer] for layer in layers]

    plt.figure(figsize=(10, 5))
    plt.bar(layers, values)
    plt.xlabel("Layer")
    plt.ylabel("Peak count")
    plt.title(f"{metric_name} peak distribution for {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric_name}_peak_histogram.png"))
    plt.close()


def run_layer_by_homonym_interaction(
    probe_fold_rows: Sequence[Dict[str, object]],
    output_dir: str,
) -> Dict[str, object]:
    result = {
        "status": "skipped",
        "reason": "statsmodels or pandas unavailable",
    }
    if not probe_fold_rows:
        save_json(os.path.join(output_dir, "layer_by_homonym_interaction.json"), result)
        return result

    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        save_json(os.path.join(output_dir, "layer_by_homonym_interaction.json"), result)
        return result

    df = pd.DataFrame(probe_fold_rows)
    interaction_model = smf.ols("probe_f1 ~ C(word) * C(layer)", data=df).fit()
    anova_table = sm.stats.anova_lm(interaction_model, typ=2)
    anova_path = os.path.join(output_dir, "layer_by_homonym_interaction_anova.csv")
    anova_table.to_csv(anova_path)

    result = {
        "status": "ok",
        "anova_path": anova_path,
        "interaction_p_value": float(anova_table.loc["C(word):C(layer)", "PR(>F)"]),
        "layer_p_value": float(anova_table.loc["C(layer)", "PR(>F)"]),
        "word_p_value": float(anova_table.loc["C(word)", "PR(>F)"]),
    }
    save_json(os.path.join(output_dir, "layer_by_homonym_interaction.json"), result)
    return result


def _analyze_word_profiles(
    word: str,
    records,
    collection,
    output_dir: str,
    random_state: int,
    cv_splits: int,
    cv_repeats: int,
) -> Dict[str, object]:
    ensure_dir(output_dir)

    labels = np.array(records["semantic_group_id"].tolist(), dtype=int)
    groups = records["family_id"].tolist()
    layers = sorted(collection.representations["target_mean"].keys())

    probe = ProbingClassifier()
    layer_rows: List[Dict[str, object]] = []
    probe_fold_rows: List[Dict[str, object]] = []

    for layer in layers:
        target_tensor = collection.representations["target_mean"][layer]
        sentence_tensor = collection.representations["sentence_mean"][layer]

        target_X, target_y = probe.prepare_data(target_tensor, labels)
        sentence_X, sentence_y = probe.prepare_data(sentence_tensor, labels)

        target_cv = probe.cross_validate(
            X=target_X,
            y=target_y,
            groups=groups,
            n_splits=cv_splits,
            n_repeats=cv_repeats,
            random_state=random_state,
        )
        sentence_cv = probe.cross_validate(
            X=sentence_X,
            y=sentence_y,
            groups=groups,
            n_splits=cv_splits,
            n_repeats=cv_repeats,
            random_state=random_state,
        )
        random_cv = probe.evaluate_shuffled_baseline(
            X=target_X,
            y=target_y,
            groups=groups,
            n_splits=cv_splits,
            n_repeats=cv_repeats,
            random_state=random_state,
        )

        gdv_value = float(compute_gdv(target_X, labels))
        gdv_separation = float(-gdv_value)

        layer_row = {
            "word": word,
            "layer": int(layer),
            "n_samples": int(len(labels)),
            "gdv": gdv_value,
            "gdv_separation": gdv_separation,
            "probe_f1_mean": float(target_cv["mean_f1"]),
            "probe_f1_std": float(target_cv["std_f1"]),
            "probe_accuracy_mean": float(target_cv["mean_accuracy"]),
            "sentence_probe_f1_mean": float(sentence_cv["mean_f1"]),
            "sentence_probe_f1_std": float(sentence_cv["std_f1"]),
            "random_label_f1_mean": float(random_cv["mean_f1"]),
            "random_label_f1_std": float(random_cv["std_f1"]),
        }
        layer_rows.append(layer_row)

        for fold_row in target_cv["fold_results"]:
            probe_fold_rows.append(
                {
                    "word": word,
                    "layer": int(layer),
                    "repeat": int(fold_row["repeat"]),
                    "split_id": int(fold_row["split_id"]),
                    "probe_f1": float(fold_row["f1"]),
                    "probe_accuracy": float(fold_row["accuracy"]),
                }
            )

    probe_scores = [float(row["probe_f1_mean"]) for row in layer_rows]
    gdv_scores = [float(row["gdv_separation"]) for row in layer_rows]
    sentence_scores = [float(row["sentence_probe_f1_mean"]) for row in layer_rows]
    random_scores = [float(row["random_label_f1_mean"]) for row in layer_rows]

    peak_probe_idx = int(np.argmax(probe_scores))
    peak_gdv_idx = int(np.argmax(gdv_scores))
    peak_probe_layer = int(layer_rows[peak_probe_idx]["layer"])
    peak_gdv_layer = int(layer_rows[peak_gdv_idx]["layer"])

    summary = {
        "word": word,
        "n_samples": int(len(labels)),
        "target_not_found_count": int(collection.metadata["target_found"].count(False)),
        "layers": [int(layer) for layer in layers],
        "peak_layer_probe": peak_probe_layer,
        "peak_layer_gdv": peak_gdv_layer,
        "top_k_probe_layers": top_k_layers(layers, probe_scores, k=3, maximize=True),
        "top_k_gdv_layers": top_k_layers(layers, gdv_scores, k=3, maximize=True),
        "probe_profile_sharpness": float(compute_profile_sharpness(probe_scores, peak_probe_idx)),
        "gdv_profile_sharpness": float(compute_profile_sharpness(gdv_scores, peak_gdv_idx)),
        "probe_peak_f1": float(probe_scores[peak_probe_idx]),
        "probe_peak_std_f1": float(layer_rows[peak_probe_idx]["probe_f1_std"]),
        "sentence_peak_f1": float(max(sentence_scores)),
        "random_label_peak_f1": float(max(random_scores)),
        "peak_margin_vs_sentence": float(probe_scores[peak_probe_idx] - sentence_scores[peak_probe_idx]),
        "peak_margin_vs_random": float(probe_scores[peak_probe_idx] - random_scores[peak_probe_idx]),
        "profiles": {
            "probe_f1_mean": {str(row["layer"]): float(row["probe_f1_mean"]) for row in layer_rows},
            "probe_f1_std": {str(row["layer"]): float(row["probe_f1_std"]) for row in layer_rows},
            "gdv_separation": {str(row["layer"]): float(row["gdv_separation"]) for row in layer_rows},
            "sentence_probe_f1_mean": {str(row["layer"]): float(row["sentence_probe_f1_mean"]) for row in layer_rows},
            "random_label_f1_mean": {str(row["layer"]): float(row["random_label_f1_mean"]) for row in layer_rows},
        },
    }

    save_csv(
        os.path.join(output_dir, f"{word}_layer_metrics.csv"),
        layer_rows,
        fieldnames=list(layer_rows[0].keys()),
    )
    save_csv(
        os.path.join(output_dir, f"{word}_probe_folds.csv"),
        probe_fold_rows,
        fieldnames=list(probe_fold_rows[0].keys()),
    )
    save_json(os.path.join(output_dir, f"{word}_summary.json"), summary)
    plot_word_profiles(word=word, layer_rows=layer_rows, output_dir=output_dir)

    return {
        "summary": summary,
        "layer_rows": layer_rows,
        "probe_fold_rows": probe_fold_rows,
    }


def _summarize_model_profiles(
    model_name: str,
    word_results: Dict[str, Dict[str, object]],
    output_dir: str,
) -> Dict[str, object]:
    peak_probe_layers = [int(result["summary"]["peak_layer_probe"]) for result in word_results.values()]
    peak_gdv_layers = [int(result["summary"]["peak_layer_gdv"]) for result in word_results.values()]

    peak_probe_rows = [
        {
            "word": word,
            "peak_layer_probe": int(result["summary"]["peak_layer_probe"]),
            "peak_layer_gdv": int(result["summary"]["peak_layer_gdv"]),
            "probe_profile_sharpness": float(result["summary"]["probe_profile_sharpness"]),
            "gdv_profile_sharpness": float(result["summary"]["gdv_profile_sharpness"]),
        }
        for word, result in sorted(word_results.items())
    ]
    save_csv(
        os.path.join(output_dir, "concept_peak_layers.csv"),
        peak_probe_rows,
        fieldnames=list(peak_probe_rows[0].keys()),
    )

    probe_band = compute_semantic_band(peak_probe_layers, band_width=3)
    gdv_band = compute_semantic_band(peak_gdv_layers, band_width=3)
    summary = {
        "model_name": model_name,
        "n_homonyms": int(len(word_results)),
        "probe_peak_layers": peak_probe_layers,
        "gdv_peak_layers": peak_gdv_layers,
        "probe_peak_variance": float(np.var(peak_probe_layers)) if peak_probe_layers else 0.0,
        "gdv_peak_variance": float(np.var(peak_gdv_layers)) if peak_gdv_layers else 0.0,
        "probe_semantic_band": probe_band,
        "gdv_semantic_band": gdv_band,
        "probe_global_layer": maybe_global_layer(peak_probe_layers),
        "gdv_global_layer": maybe_global_layer(peak_gdv_layers),
        "words": {word: result["summary"] for word, result in sorted(word_results.items())},
    }
    save_json(os.path.join(output_dir, "model_summary.json"), summary)
    plot_peak_histogram(model_name, peak_probe_layers, "probe", output_dir)
    plot_peak_histogram(model_name, peak_gdv_layers, "gdv", output_dir)
    return summary


def run_distributed_semantic_profile_experiment(
    df,
    model_name: str,
    model_type: str = "default",
    base_dir: str = "results",
    batch_size: int = 8,
    cv_splits: int = 5,
    cv_repeats: int = 3,
    random_state: int = 42,
) -> Dict[str, object]:
    model_output_dir = os.path.join(base_dir, "distributed_profiles", sanitize_model_name(model_name))
    ensure_dir(model_output_dir)

    model, tokenizer = load_model_and_tokenizer(model_name, model_type=model_type)
    df_flat = flatten_dataframe(df)

    word_results: Dict[str, Dict[str, object]] = {}
    all_probe_fold_rows: List[Dict[str, object]] = []

    for word in sorted(df_flat["word"].unique()):
        word_records = df_flat[df_flat["word"] == word].reset_index(drop=True)
        word_output_dir = os.path.join(model_output_dir, word)
        ensure_dir(word_output_dir)

        collection = collect_target_span_representations(
            model=model,
            tokenizer=tokenizer,
            texts=word_records["sentence"].tolist(),
            targets=word_records["word"].tolist(),
            batch_size=batch_size,
            representation_kinds=("target_mean", "target_last", "sentence_mean"),
        )

        activation_bundle = {
            "metadata": collection.metadata,
            "representations": {
                rep_name: {int(layer): tensor for layer, tensor in layer_dict.items()}
                for rep_name, layer_dict in collection.representations.items()
            },
        }
        activation_path = os.path.join(word_output_dir, f"{word}_representations.pt")
        torch.save(activation_bundle, activation_path)

        analysis = _analyze_word_profiles(
            word=word,
            records=word_records,
            collection=collection,
            output_dir=word_output_dir,
            random_state=random_state,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
        )
        analysis["summary"]["activation_path"] = activation_path
        word_results[word] = analysis
        all_probe_fold_rows.extend(analysis["probe_fold_rows"])

    interaction_summary = run_layer_by_homonym_interaction(
        probe_fold_rows=all_probe_fold_rows,
        output_dir=model_output_dir,
    )
    model_summary = _summarize_model_profiles(
        model_name=model_name,
        word_results=word_results,
        output_dir=model_output_dir,
    )
    model_summary["interaction_test"] = interaction_summary
    save_json(os.path.join(model_output_dir, "model_summary.json"), model_summary)

    return {
        "model_name": model_name,
        "output_dir": model_output_dir,
        "word_results": word_results,
        "model_summary": model_summary,
    }


def _profile_dict_to_arrays(profile_dict: Dict[str, float]) -> tuple[List[int], List[float]]:
    layers = sorted(int(layer) for layer in profile_dict.keys())
    scores = [float(profile_dict[str(layer)]) for layer in layers]
    return layers, scores


def _save_matrix_csv(path: str, matrix: np.ndarray, row_labels: Sequence[int], col_labels: Sequence[int]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["layer"] + [str(label) for label in col_labels])
        for row_label, row_values in zip(row_labels, matrix):
            writer.writerow([row_label] + [float(value) for value in row_values])


def run_across_model_profile_comparison(
    experiment_results: Dict[str, Dict[str, object]],
    base_dir: str = "results",
    interpolation_points: int = 64,
) -> Dict[str, object]:
    output_dir = os.path.join(base_dir, "cross_model_profiles")
    ensure_dir(output_dir)

    pair_rows: List[Dict[str, object]] = []
    aggregate_rows: List[Dict[str, object]] = []

    for model_a, model_b in combinations(sorted(experiment_results.keys()), 2):
        result_a = experiment_results[model_a]
        result_b = experiment_results[model_b]
        common_words = sorted(set(result_a["word_results"].keys()) & set(result_b["word_results"].keys()))
        pair_dir = os.path.join(output_dir, f"{sanitize_model_name(model_a)}__vs__{sanitize_model_name(model_b)}")
        ensure_dir(pair_dir)

        pair_word_rows: List[Dict[str, object]] = []
        for word in common_words:
            word_summary_a = result_a["word_results"][word]["summary"]
            word_summary_b = result_b["word_results"][word]["summary"]

            probe_layers_a, probe_scores_a = _profile_dict_to_arrays(word_summary_a["profiles"]["probe_f1_mean"])
            probe_layers_b, probe_scores_b = _profile_dict_to_arrays(word_summary_b["profiles"]["probe_f1_mean"])
            gdv_layers_a, gdv_scores_a = _profile_dict_to_arrays(word_summary_a["profiles"]["gdv_separation"])
            gdv_layers_b, gdv_scores_b = _profile_dict_to_arrays(word_summary_b["profiles"]["gdv_separation"])

            interpolated_probe_a = interpolate_profile(probe_layers_a, probe_scores_a, num_points=interpolation_points)
            interpolated_probe_b = interpolate_profile(probe_layers_b, probe_scores_b, num_points=interpolation_points)
            interpolated_gdv_a = interpolate_profile(gdv_layers_a, gdv_scores_a, num_points=interpolation_points)
            interpolated_gdv_b = interpolate_profile(gdv_layers_b, gdv_scores_b, num_points=interpolation_points)

            activation_path_a = word_summary_a["activation_path"]
            activation_path_b = word_summary_b["activation_path"]
            activations_a = torch.load(activation_path_a, map_location="cpu")
            activations_b = torch.load(activation_path_b, map_location="cpu")
            target_mean_a = activations_a["representations"]["target_mean"]
            target_mean_b = activations_b["representations"]["target_mean"]
            cka_layers_a = sorted(int(layer) for layer in target_mean_a.keys())
            cka_layers_b = sorted(int(layer) for layer in target_mean_b.keys())

            cka_matrix = np.zeros((len(cka_layers_a), len(cka_layers_b)), dtype=float)
            for row_index, layer_a in enumerate(cka_layers_a):
                act_a = target_mean_a[layer_a].detach().cpu().numpy()
                for col_index, layer_b in enumerate(cka_layers_b):
                    act_b = target_mean_b[layer_b].detach().cpu().numpy()
                    cka_matrix[row_index, col_index] = linear_cka(act_a, act_b)

            best_cka_index = np.unravel_index(np.argmax(cka_matrix), cka_matrix.shape)
            best_cka_layer_a = int(cka_layers_a[best_cka_index[0]])
            best_cka_layer_b = int(cka_layers_b[best_cka_index[1]])
            best_cka_value = float(cka_matrix[best_cka_index])

            matrix_path = os.path.join(pair_dir, f"{word}_cka_matrix.csv")
            _save_matrix_csv(matrix_path, cka_matrix, cka_layers_a, cka_layers_b)

            probe_peak_distance = abs(
                relative_peak_position(probe_layers_a, int(word_summary_a["peak_layer_probe"]))
                - relative_peak_position(probe_layers_b, int(word_summary_b["peak_layer_probe"]))
            )
            gdv_peak_distance = abs(
                relative_peak_position(gdv_layers_a, int(word_summary_a["peak_layer_gdv"]))
                - relative_peak_position(gdv_layers_b, int(word_summary_b["peak_layer_gdv"]))
            )

            row = {
                "model_a": model_a,
                "model_b": model_b,
                "word": word,
                "probe_profile_corr": float(safe_pearsonr(interpolated_probe_a, interpolated_probe_b)),
                "gdv_profile_corr": float(safe_pearsonr(interpolated_gdv_a, interpolated_gdv_b)),
                "probe_peak_distance": float(probe_peak_distance),
                "gdv_peak_distance": float(gdv_peak_distance),
                "matched_probe_peak_window": bool(probe_peak_distance <= 0.10),
                "matched_gdv_peak_window": bool(gdv_peak_distance <= 0.10),
                "best_cka_layer_a": best_cka_layer_a,
                "best_cka_layer_b": best_cka_layer_b,
                "best_cka_value": best_cka_value,
                "cka_matrix_path": matrix_path,
            }
            pair_rows.append(row)
            pair_word_rows.append(row)

        if pair_word_rows:
            aggregate_row = {
                "model_a": model_a,
                "model_b": model_b,
                "n_common_homonyms": len(pair_word_rows),
                "mean_probe_profile_corr": float(np.mean([row["probe_profile_corr"] for row in pair_word_rows])),
                "mean_gdv_profile_corr": float(np.mean([row["gdv_profile_corr"] for row in pair_word_rows])),
                "mean_probe_peak_distance": float(np.mean([row["probe_peak_distance"] for row in pair_word_rows])),
                "mean_gdv_peak_distance": float(np.mean([row["gdv_peak_distance"] for row in pair_word_rows])),
                "probe_peak_window_match_rate": float(np.mean([row["matched_probe_peak_window"] for row in pair_word_rows])),
                "gdv_peak_window_match_rate": float(np.mean([row["matched_gdv_peak_window"] for row in pair_word_rows])),
                "mean_best_cka_value": float(np.mean([row["best_cka_value"] for row in pair_word_rows])),
            }
            aggregate_rows.append(aggregate_row)

    if pair_rows:
        save_csv(
            os.path.join(output_dir, "cross_model_word_level.csv"),
            pair_rows,
            fieldnames=list(pair_rows[0].keys()),
        )
    if aggregate_rows:
        save_csv(
            os.path.join(output_dir, "cross_model_summary.csv"),
            aggregate_rows,
            fieldnames=list(aggregate_rows[0].keys()),
        )

    summary = {
        "output_dir": output_dir,
        "pair_count": len(aggregate_rows),
        "word_level_rows": len(pair_rows),
    }
    save_json(os.path.join(output_dir, "summary.json"), summary)
    return summary
