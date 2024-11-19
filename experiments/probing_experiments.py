import csv
import random
import os
from models import get_activations
from probing import ProbingClassifier
from data import DatasetPreprocessor

def run_layer_identification_experiment(model, tokenizer, dataset_name, split, results_dir="results"):
    """
    Run the layer identification experiment.

    Args:
        model: Pre-trained transformer model.
        tokenizer: Tokenizer for the model.
        dataset_name (str): Name of the dataset (e.g., "wic").
        split (str): Dataset split to use (e.g., "train").
        results_dir (str): Directory to save results.
    """
    # Create preprocessor dynamically
    preprocessor = DatasetPreprocessor(tokenizer, dataset_name)
    texts, labels = preprocessor.load_and_prepare(split)
    #texts, labels = texts[:100], labels[:100]  # Use a subset for quick testing

    # Prepare results storage
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"H1-{model.name_or_path.replace('/', '_')}-{dataset_name.capitalize()}-results.csv")

    # Open results file
    with open(results_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Layer", "Activations Shape", "Accuracy", "F1 Score"])

        # Iterate through all layers
        probing = ProbingClassifier()
        for layer_idx in range(model.config.num_hidden_layers):
            try:
                # Extract activations for the current layer
                activations = get_activations(model, tokenizer, texts, layer_indices=[layer_idx])
                layer_activations = activations[layer_idx]

                # Skip if activations have an invalid shape
                if layer_activations.size(0) != len(labels):
                    print(f"Skipping Layer {layer_idx} due to shape mismatch: {layer_activations.shape}")
                    continue

                # Prepare data for the probing classifier
                X, y = probing.prepare_data(layer_activations, labels)

                # Train and evaluate the probing classifier
                accuracy, f1 = probing.train(X, y)
                print(f"Layer {layer_idx} Activations Shape: {layer_activations.shape}")
                print(f"Probing Classifier Results for Layer {layer_idx}:")
                print(f"Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}")

                # Save results
                writer.writerow([layer_idx, layer_activations.shape, accuracy, f1])

            except Exception as e:
                print(f"Error processing Layer {layer_idx}: {e}")
