import csv
import os
import logging
import traceback
import torch
import gc
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
    logging.info(f"Running layer identification experiment for model: {model.name_or_path}")

    preprocessor = DatasetPreprocessor(tokenizer, dataset_name)
    texts, labels = preprocessor.load_and_prepare(split)
    logging.debug(f"Loaded and prepared dataset: {dataset_name}, split: {split}")

    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"H1-{model.name_or_path.replace('/', '_')}-{dataset_name.capitalize()}-results.csv")

    with open(results_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Layer", "Activations Shape", "Accuracy", "F1 Score"])

        probing = ProbingClassifier()
        for layer_idx in range(model.config.num_hidden_layers):
            try:
                logging.debug(f"Processing layer {layer_idx}")
                activations = get_activations(model, tokenizer, texts, layer_indices=[layer_idx])
                
                # Check if activations for the current layer were collected
                if layer_idx not in activations:
                    logging.error(f"Activations for layer {layer_idx} were not collected. Available keys: {list(activations.keys())}")
                    continue  # Skip this layer
                
                layer_activations = activations[layer_idx]

                # If activations are token-level (3D) then pool them to obtain one vector per example.
                if len(layer_activations.shape) == 3:
                    original_shape = layer_activations.shape
                    layer_activations = layer_activations.mean(dim=1)
                    logging.debug(f"Applied mean pooling on layer {layer_idx}: {original_shape} -> {layer_activations.shape}")

                # Check for shape mismatch (i.e. one activation vector per label)
                if layer_activations.size(0) != len(labels):
                    logging.warning(f"Skipping Layer {layer_idx} due to shape mismatch: {layer_activations.shape} vs. expected {len(labels)}")
                    continue

                X, y = probing.prepare_data(layer_activations, labels)
                accuracy, f1 = probing.train(X, y)
                logging.info(f"Layer {layer_idx} - Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}")
                writer.writerow([layer_idx, layer_activations.shape, accuracy, f1])
            
            except Exception as e:
                logging.error(f"Error processing Layer {layer_idx}: {e}")
                logging.debug(traceback.format_exc())
            finally:
                if 'activations' in locals():
                    try:
                        del activations
                    except Exception as ex:
                        logging.debug(f"Failed to delete 'activations' for layer {layer_idx}: {ex}")
                if 'layer_activations' in locals():
                    try:
                        del layer_activations
                    except Exception as ex:
                        logging.debug(f"Failed to delete 'layer_activations' for layer {layer_idx}: {ex}")
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logging.debug(f"Finished cleanup for layer {layer_idx}")
