import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import torch 


class ProbingClassifier:
    """
    A class to handle training and evaluating a probing classifier on activations.
    """

    def __init__(self, classifier=None):
        """
        Initialize the ProbingClassifier.

        Args:
            classifier: A scikit-learn style classifier. Defaults to LogisticRegression.
        """
        self.classifier = classifier or LogisticRegression(max_iter=1000)
        self.results = {}

    def prepare_data(self, activations, labels):
        """
        Prepare activations and labels for training.
        Args:
            activations (torch.Tensor): Tensor of shape (N, features...).
            labels (list or np.ndarray): List or array of labels of size N.
        Returns:
            X (np.ndarray): Flattened activations.
            y (np.ndarray): Labels.
        """
        # Convert to float32 if needed
        activations = activations.to(torch.float32)
        X = activations.view(activations.size(0), -1).cpu().numpy()
        y = np.array(labels)
        return X, y


    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the probing classifier on the data.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels.
            test_size (float): Fraction of data to use for testing.
            random_state (int): Random seed for reproducibility.

        Returns:
            accuracy (float): Accuracy on the test set.
        """
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.classifier.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        self.results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "classification_report": classification_report(y_test, y_pred, zero_division=1),
        }
        return accuracy, f1

    def get_results(self):
        """
        Retrieve the stored results from the last training run.

        Returns:
            dict: A dictionary containing accuracy, F1 score, and a classification report.
        """
        return self.results

    def save_results(self, path):
        """
        Save the results to a file.

        Args:
            path (str): Path to save the results.
        """
        with open(path, 'w') as f:
            for key, value in self.results.items():
                f.write(f"{key}:\n{value}\n\n")

    @staticmethod
    def plot_accuracy_by_layer(layer_indices, accuracies, save_path=None):
        """
        Plot accuracy by layer.

        Args:
            layer_indices (list): List of layer indices.
            accuracies (list): List of accuracies for each layer.
            save_path (str): Path to save the plot. If None, just display it.
        """
        plt.figure()
        plt.plot(layer_indices, accuracies, marker='o')
        plt.title("Probing Classifier Accuracy by Layer")
        plt.xlabel("Layer Index")
        plt.ylabel("Accuracy")
        plt.grid()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
