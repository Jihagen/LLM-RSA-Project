import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on sklearn version
    StratifiedGroupKFold = None


class ProbingClassifier:
    """
    Train and evaluate linear probes on activation vectors.
    """

    def __init__(self, classifier=None):
        self.classifier = classifier or LogisticRegression(max_iter=1000)
        self.results: Dict[str, object] = {}

    def prepare_data(self, activations, labels):
        activations = activations.to(torch.float32)
        X = activations.view(activations.size(0), -1).cpu().numpy()
        y = np.array(labels)
        return X, y

    def train(self, X, y, test_size=0.2, return_predictions=False, random_state=None):
        if random_state is None:
            random_state = random.randint(0, 100000)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        classifier = clone(self.classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)
        self.results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "classification_report": classification_report(y_test, y_pred, zero_division=1),
        }

        if return_predictions:
            return accuracy, f1, y_pred
        return accuracy, f1

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[Sequence[str]] = None,
        n_splits: int = 5,
        n_repeats: int = 3,
        random_state: int = 42,
        return_fold_predictions: bool = False,
    ) -> Dict[str, object]:
        fold_results: List[Dict[str, object]] = []
        prediction_rows: List[Dict[str, object]] = []

        groups_array = np.array(groups) if groups is not None else None
        split_id = 0

        for repeat in range(n_repeats):
            splitter = self._build_splitter(
                y=y,
                groups=groups_array,
                n_splits=n_splits,
                random_state=random_state + repeat,
            )

            split_iter = (
                splitter.split(X, y, groups_array)
                if groups_array is not None and self._uses_group_splitter(splitter)
                else splitter.split(X, y)
            )

            for train_idx, test_idx in split_iter:
                classifier = clone(self.classifier)
                classifier.fit(X[train_idx], y[train_idx])
                y_pred = classifier.predict(X[test_idx])

                accuracy = float(accuracy_score(y[test_idx], y_pred))
                f1 = float(f1_score(y[test_idx], y_pred, average="weighted", zero_division=1))
                fold_result = {
                    "split_id": split_id,
                    "repeat": repeat,
                    "accuracy": accuracy,
                    "f1": f1,
                    "support": int(len(test_idx)),
                }
                fold_results.append(fold_result)

                if return_fold_predictions:
                    for sample_index, gold, pred in zip(test_idx, y[test_idx], y_pred):
                        prediction_rows.append(
                            {
                                "split_id": split_id,
                                "repeat": repeat,
                                "sample_index": int(sample_index),
                                "gold": int(gold),
                                "prediction": int(pred),
                            }
                        )

                split_id += 1

        accuracies = [row["accuracy"] for row in fold_results]
        f1_scores = [row["f1"] for row in fold_results]
        summary = {
            "n_splits_total": len(fold_results),
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "std_accuracy": float(np.std(accuracies)) if accuracies else 0.0,
            "mean_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "std_f1": float(np.std(f1_scores)) if f1_scores else 0.0,
            "fold_results": fold_results,
        }
        if return_fold_predictions:
            summary["predictions"] = prediction_rows
        return summary

    def evaluate_shuffled_baseline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[Sequence[str]] = None,
        n_splits: int = 5,
        n_repeats: int = 3,
        random_state: int = 42,
    ) -> Dict[str, object]:
        shuffled_rng = np.random.default_rng(random_state)
        shuffled_y = shuffled_rng.permutation(np.array(y))
        return self.cross_validate(
            X=X,
            y=shuffled_y,
            groups=groups,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    def _build_splitter(
        self,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        n_splits: int,
        random_state: int,
    ):
        if self._can_use_group_cv(y, groups, n_splits):
            return StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

    @staticmethod
    def _uses_group_splitter(splitter) -> bool:
        return StratifiedGroupKFold is not None and isinstance(splitter, StratifiedGroupKFold)

    @staticmethod
    def _can_use_group_cv(
        y: np.ndarray,
        groups: Optional[np.ndarray],
        n_splits: int,
    ) -> bool:
        if groups is None or StratifiedGroupKFold is None:
            return False

        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            return False

        labels = np.unique(y)
        for label in labels:
            label_groups = np.unique(groups[y == label])
            if len(label_groups) < 2:
                return False
        return True

    def get_results(self):
        return self.results

    def save_results(self, path):
        with open(path, "w") as f:
            for key, value in self.results.items():
                f.write(f"{key}:\n{value}\n\n")

    @staticmethod
    def plot_accuracy_by_layer(layer_indices, accuracies, save_path=None):
        plt.figure()
        plt.plot(layer_indices, accuracies, marker="o")
        plt.title("Probing Classifier Accuracy by Layer")
        plt.xlabel("Layer Index")
        plt.ylabel("Accuracy")
        plt.grid()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
