# Main functions for each experiment

from models.activations import load_model_and_tokenizer, get_activations, compute_rdm
from visualisations.visualisation import plot_rdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import numpy as np


def compare_activations_per_layer(models, texts):
    for model_name in models:
       compare_activations_per_layer_modelSpec(model_name, texts)


def compare_activations_per_layer_modelSpec(model_name, texts):
    model, tokenizer = load_model_and_tokenizer(model_name)
    all_activations = get_activations(model, tokenizer, texts)

    for layer, activation in all_activations.items():
        rdm = compute_rdm(activation)
        plot_rdm(rdm, f"Layer {layer} Across Samples", method="MDS")


def compare_activations_per_sample(models, samples):
    for sample in samples:
        activations_by_model = []
        model_labels = []
        
        # Gather activations for each model
        for model_name in models:
            model, tokenizer = load_model_and_tokenizer(model_name)
            activations = get_activations(model, tokenizer, [sample])
            first_layer_activation = activations[next(iter(activations))]
            
            # Aggregate activations by taking the mean across tokens (dimension 1)
            aggregated_activation = first_layer_activation.mean(dim=1)
            activations_by_model.append(aggregated_activation)
            model_labels.append(model_name)
        
        # Concatenate activations across models (should now have matching dimensions)
        combined_activations = torch.cat(activations_by_model, dim=0)
        combined_rdm = compute_rdm(combined_activations)
        
        # Perform MDS on the combined RDM
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        points_2d = mds.fit_transform(combined_rdm)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        
        # Plot each model's points in different colors
        colors = plt.cm.get_cmap('tab10', len(models))
        start_idx = 0
        for i, model_name in enumerate(models):
            num_points = activations_by_model[i].shape[0]
            plt.scatter(points_2d[start_idx:start_idx + num_points, 0], 
                        points_2d[start_idx:start_idx + num_points, 1], 
                        s=100, color=colors(i), label=model_name, edgecolor="k")
            start_idx += num_points

        plt.title(f"Combined MDS Plot for Sample: '{sample}' Across Models")
        plt.xlabel("MDS Dimension 1")
        plt.ylabel("MDS Dimension 2")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()





def compare_activations_across_models_by_position(models, texts, reduced_dim=256):
    """
    Compare activations per layer across multiple models by matching layers by position
    and reducing feature dimensions to a fixed size using PCA, while skipping non-comparable layers.
    
    Parameters:
        models (list of str): List of model names to compare.
        texts (list of str): List of input texts to pass to each model.
        reduced_dim (int): Target dimensionality for PCA reduction.
    """
    model_activations = {}
    
    for model_name in models:
        model, tokenizer = load_model_and_tokenizer(model_name)
        activations = get_activations(model, tokenizer, texts)
        model_activations[model_name] = activations

    # Determine the minimum number of layers across all models
    min_layer_count = min(len(activations) for activations in model_activations.values())
    
    # Define keywords to identify non-comparable layers, e.g., embedding layers
    non_comparable_keywords = ["embedding", "wpe", "wte", "position", "token_type"]

    # Iterate over each layer position up to the minimum layer count
    for layer_index in range(min_layer_count):
        activations_by_model = []
        model_labels = []
        sample_labels = []
        
        for model_name in models:
            layer_name = list(model_activations[model_name].keys())[layer_index]
            if any(keyword in layer_name.lower() for keyword in non_comparable_keywords):
                print(f"Skipping non-comparable layer {layer_name} for model {model_name}")
                continue
            
            print(f"Processing layer {layer_name} for model {model_name}")
            activation = model_activations[model_name][layer_name]
            aggregated_activation = activation.mean(dim=1)  # Aggregate across tokens

            # Only apply PCA if there are enough samples and features
            if aggregated_activation.shape[0] > 1 and aggregated_activation.shape[1] > 1:
                max_pca_dim = min(reduced_dim, aggregated_activation.shape[1], aggregated_activation.shape[0])
                pca = PCA(n_components=max_pca_dim)
                reduced_activation = torch.tensor(pca.fit_transform(aggregated_activation.cpu().numpy()))
            else:
                reduced_activation = aggregated_activation

            activations_by_model.append(reduced_activation)
            model_labels.extend([model_name] * reduced_activation.shape[0])
            sample_labels.extend([f"{i}" for i in range(reduced_activation.shape[0])])

        # Ensure we have valid activations for this layer across models
        if len(activations_by_model) < len(models):
            print(f"Skipping layer {layer_name} due to insufficient comparable data across models.")
            continue

        # Concatenate activations across models for the current layer
        combined_activations = torch.cat(activations_by_model, dim=0)
        combined_rdm = compute_rdm(combined_activations)
        
        # Perform MDS on the combined RDM for visualization
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        points_2d = mds.fit_transform(combined_rdm)

        # Plot the combined RDM for the current layer
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10', len(models))
        
        # Plot each model's points with distinct colors and add sample labels
        start_idx = 0
        for i, model_name in enumerate(models):
            num_points = activations_by_model[i].shape[0]
            plt.scatter(points_2d[start_idx:start_idx + num_points, 0], 
                        points_2d[start_idx:start_idx + num_points, 1], 
                        s=100, color=colors(i), label=model_name, edgecolor="k")
            
            # Add labels for each sample point
            for j in range(num_points):
                plt.text(points_2d[start_idx + j, 0], points_2d[start_idx + j, 1], 
                         sample_labels[start_idx + j], fontsize=8, ha="right")
            
            start_idx += num_points

        plt.title(f"Combined MDS Plot for Layer Position {layer_index} Across Models")
        plt.xlabel("MDS Dimension 1")
        plt.ylabel("MDS Dimension 2")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
