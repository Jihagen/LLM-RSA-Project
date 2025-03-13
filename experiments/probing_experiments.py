import csv
import os
import logging
import traceback
import torch
import gc
import pickle
from models import get_activations
from probing import ProbingClassifier
from sklearn.linear_model import LogisticRegression
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

    # Use the updated preprocessor that returns text pairs.
    preprocessor = DatasetPreprocessor(tokenizer, dataset_name)
    texts1, texts2, labels = preprocessor.load_and_prepare(split)
    logging.debug(f"Loaded and prepared dataset: {dataset_name}, split: {split}")

    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"H1-{model.name_or_path.replace('/', '_')}-{dataset_name.capitalize()}-results.csv")
    
    # You may also want to save best-run predictions per layer into a separate file or directory.
    best_preds_dir = os.path.join(results_dir, f"{model.name_or_path.replace('/', '_')}_H1_best_predictions")
    os.makedirs(best_preds_dir, exist_ok=True)

    with open(results_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header with placeholders for five runs.
        writer.writerow(["Layer", "Activations Shape", 
                         "Run1 Accuracy", "Run1 F1", 
                         "Run2 Accuracy", "Run2 F1",
                         "Run3 Accuracy", "Run3 F1",
                         "Run4 Accuracy", "Run4 F1",
                         "Run5 Accuracy", "Run5 F1"])

        # Loop over each layer in the model.
        for layer_idx in range(model.config.num_hidden_layers):
            try:
                logging.debug(f"Processing layer {layer_idx}")

                # IMPORTANT: Pass texts1 and texts2 to get_activations.
                activations = get_activations(model, tokenizer, texts1, texts2, layer_indices=[layer_idx])
                
                # Check that activations for the current layer were collected.
                if layer_idx not in activations:
                    logging.error(f"Activations for layer {layer_idx} were not collected. Available keys: {list(activations.keys())}")
                    continue  # Skip this layer
                
                # Assume get_activations returns a tuple: (act_text1, act_text2)
                layer_activations_pair = activations[layer_idx]
                act_text1, act_text2 = layer_activations_pair

                # If activations are token-level (3D) then pool them (using mean pooling here).
                if len(act_text1.shape) == 3:
                    original_shape = act_text1.shape
                    act_text1 = act_text1.mean(dim=1)
                    logging.debug(f"Applied mean pooling on layer {layer_idx} for text1: {original_shape} -> {act_text1.shape}")
                if len(act_text2.shape) == 3:
                    original_shape = act_text2.shape
                    act_text2 = act_text2.mean(dim=1)
                    logging.debug(f"Applied mean pooling on layer {layer_idx} for text2: {original_shape} -> {act_text2.shape}")

                # Combine the two activations per sample (concatenation along the feature dimension).
                # For each sample, this yields a vector of shape [2 * hidden_dim].
                if act_text1.size(0) != len(labels) or act_text2.size(0) != len(labels):
                    logging.warning(f"Skipping layer {layer_idx} due to shape mismatch in activations vs. labels.")
                    continue

                combined_activations = torch.cat([act_text1, act_text2], dim=1)
                logging.debug(f"Combined activations shape for layer {layer_idx}: {combined_activations.shape}")

                # Now, train multiple probing classifiers.
                probing = ProbingClassifier(classifier=LogisticRegression(max_iter=1000))
                num_runs = 5
                all_accuracies = []
                all_f1s = []
                best_accuracy = -1
                best_predictions = None

                # Train the classifier for num_runs passes.
                for run in range(num_runs):
                    X, y = probing.prepare_data(combined_activations, labels)
                    # Here, assume probing.train returns (accuracy, f1, predictions)
                    accuracy, f1, predictions = probing.train(X, y, return_predictions=True)
                    all_accuracies.append(accuracy)
                    all_f1s.append(f1)
                    logging.info(f"Layer {layer_idx}, Run {run}: Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
                    # Keep the predictions from the best run (using accuracy as the metric).
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_predictions = predictions

                # Save best run predictions to a file (one per layer).
                preds_file = os.path.join(best_preds_dir, f"layer_{layer_idx}_best_predictions.pkl")
                with open(preds_file, "wb") as pf:
                    pickle.dump(best_predictions, pf)

                # Write one row per layer with results from each run.
                writer.writerow([layer_idx, combined_activations.shape,
                                 all_accuracies[0], all_f1s[0],
                                 all_accuracies[1], all_f1s[1],
                                 all_accuracies[2], all_f1s[2],
                                 all_accuracies[3], all_f1s[3],
                                 all_accuracies[4], all_f1s[4]])
            
            except Exception as e:
                logging.error(f"Error processing layer {layer_idx}: {e}")
                logging.debug(traceback.format_exc())
            finally:
                # Cleanup: delete large variables and collect garbage.
                if 'activations' in locals():
                    try:
                        del activations
                    except Exception as ex:
                        logging.debug(f"Failed to delete 'activations' for layer {layer_idx}: {ex}")
                if 'layer_activations_pair' in locals():
                    try:
                        del layer_activations_pair
                    except Exception as ex:
                        logging.debug(f"Failed to delete 'layer_activations_pair' for layer {layer_idx}: {ex}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logging.debug(f"Finished cleanup for layer {layer_idx}")
