import csv
import logging
import os
from typing import Dict, List

from sklearn.linear_model import LogisticRegression

from data import DatasetPreprocessor
from models import get_activations
from probing import ProbingClassifier


def run_layer_identification_experiment(model, tokenizer, dataset_name, split, results_dir="results"):
    """
    Secondary validation path on paired-text datasets such as WiC.

    This is no longer the primary H1 pipeline; it remains useful as a
    task-level validation set once the homonym-first pipeline is stable.
    """

    logging.info("Running layer identification experiment for model: %s", model.name_or_path)

    preprocessor = DatasetPreprocessor(tokenizer, dataset_name)
    texts1, texts2, labels = preprocessor.load_and_prepare(split)
    total_layers = model.config.num_hidden_layers + 1

    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(
        results_dir,
        f"H1-{model.name_or_path.replace('/', '_')}-{dataset_name.capitalize()}-results.csv",
    )

    classifier = ProbingClassifier(classifier=LogisticRegression(max_iter=1000))
    rows: List[Dict[str, object]] = []

    for layer_idx in range(total_layers):
        logging.info("Processing paired-text validation layer %s", layer_idx)
        activations = get_activations(
            model=model,
            tokenizer=tokenizer,
            texts1=texts1,
            texts2=texts2,
            layer_indices=[layer_idx],
            batch_size=8,
        )
        act_text1, act_text2 = activations[layer_idx]
        combined_activations = act_text1 - act_text2
        X, y = classifier.prepare_data(combined_activations, labels)
        cv_results = classifier.cross_validate(
            X=X,
            y=y,
            groups=None,
            n_splits=5,
            n_repeats=3,
            random_state=42,
        )

        row = {
            "Layer": layer_idx,
            "Activations Shape": str(tuple(combined_activations.shape)),
            "Mean Accuracy": cv_results["mean_accuracy"],
            "Std Accuracy": cv_results["std_accuracy"],
            "Mean F1": cv_results["mean_f1"],
            "Std F1": cv_results["std_f1"],
            "Num Folds": cv_results["n_splits_total"],
        }
        for fold_index, fold_result in enumerate(cv_results["fold_results"], start=1):
            row[f"Run{fold_index} Accuracy"] = float(fold_result["accuracy"])
            row[f"Run{fold_index} F1"] = float(fold_result["f1"])
        rows.append(row)

    with open(results_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Saved paired-text validation results to %s", results_file)
