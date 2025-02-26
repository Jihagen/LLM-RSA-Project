import csv
import random
import os
import logging
import traceback
from models import get_activations
from probing import ProbingClassifier
from data import DatasetPreprocessor
import torch

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
    logging.info(f"Running layer identification experiment for model: {model.name_or_path}")

    # Create preprocessor dynamically
    preprocessor = DatasetPreprocessor(tokenizer, dataset_name)
    texts, labels = preprocessor.load_and_prepare(split)
    logging.debug(f"Loaded and prepared dataset: {dataset_name}, split: {split}")

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
                logging.debug(f"Processing layer {layer_idx}")
                activations = get_activations(model, tokenizer, texts, layer_indices=[layer_idx])
                layer_activations = activations[layer_idx]

                if layer_activations.size(0) != len(labels):
                    logging.warning(f"Skipping Layer {layer_idx} due to shape mismatch: {layer_activations.shape}")
                    continue

                X, y = probing.prepare_data(layer_activations, labels)
                accuracy, f1 = probing.train(X, y)
                logging.info(f"Layer {layer_idx} - Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}")
                writer.writerow([layer_idx, layer_activations.shape, accuracy, f1])
            
            except Exception as e:
                logging.error(f"Error processing Layer {layer_idx}: {e}")
                logging.debug(traceback.format_exc())
            finally:
                # Delete activations and force garbage collection after each layer
                if 'activations' in locals():
                    del activations, layer_activations
                import gc
                gc.collect()
                torch.cuda.empty_cache()
