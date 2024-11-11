# Main functions for each experiment

from activations import load_model_and_tokenizer, get_activations, compute_rdm
from visualisation import plot_rdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
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


def compare_activations_across_models_variable_layers(models, texts, layer_mapping=None):
    """
    Compare activations per layer across models with flexible layer names.
    
    Parameters:
        models (list of str): List of model names to compare.
        texts (list of str): List of input texts to pass to each model.
        layer_mapping (dict): Optional dictionary to map layer names across models, e.g.,
                              {'bert-layer-1': 'gpt2-layer-1'}.
    """
    # Gather activations for each model based on provided layer mapping
    model_activations = {}

    for model_name in models:
        model, tokenizer = load_model_and_tokenizer(model_name)
        activations = get_activations(model, tokenizer, texts)
        
        # Adjust layer names if a mapping is provided
        if layer_mapping:
            activations = {layer_mapping.get(layer, layer): act for layer, act in activations.items()}
        
        model_activations[model_name] = activations

    # Determine the layers to compare, based on intersection if no mapping provided
    if not layer_mapping:
        common_layers = set.intersection(*(set(activations.keys()) for activations in model_activations.values()))
    else:
        common_layers = set(layer_mapping.values())

    # Iterate over each common layer to gather and plot activations across models
    for layer in common_layers:
        activations_by_model = []
        model_labels = []

        # Aggregate activations per model for the current layer
        for model_name in models:
            if layer in model_activations[model_name]:
                activation = model_activations[model_name][layer]
                
                # Aggregate across tokens (e.g., by mean) to get a single vector per sample
                aggregated_activation = activation.mean(dim=1)
                activations_by_model.append(aggregated_activation)
                model_labels.extend([model_name] * aggregated_activation.shape[0])

        # Concatenate activations across models for the current layer
        combined_activations = torch.cat(activations_by_model, dim=0)
        combined_rdm = compute_rdm(combined_activations)
        
        # Perform MDS on the combined RDM for visualization
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        points_2d = mds.fit_transform(combined_rdm)

        # Plot the combined RDM for the current layer
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10', len(models))

        # Plot each model's points with distinct colors
        start_idx = 0
        for i, model_name in enumerate(models):
            num_points = activations_by_model[i].shape[0]
            plt.scatter(points_2d[start_idx:start_idx + num_points, 0], 
                        points_2d[start_idx:start_idx + num_points, 1], 
                        s=100, color=colors(i), label=model_name, edgecolor="k")
            start_idx += num_points

        plt.title(f"Combined MDS Plot for Layer: '{layer}' Across Models")
        plt.xlabel("MDS Dimension 1")
        plt.ylabel("MDS Dimension 2")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


def compare_activations_across_models_by_position(models, texts):
    """
    Compare activations per layer across multiple models by matching layers by position.
    
    Parameters:
        models (list of str): List of model names to compare.
        texts (list of str): List of input texts to pass to each model.
    """
    # Gather activations for each model
    model_activations = {}
    for model_name in models:
        model, tokenizer = load_model_and_tokenizer(model_name)
        activations = get_activations(model, tokenizer, texts)
        model_activations[model_name] = activations

    # Determine the minimum number of layers across all models
    min_layer_count = min(len(activations) for activations in model_activations.values())
    
    # Iterate over each layer position up to the minimum layer count
    for layer_index in range(min_layer_count):
        activations_by_model = []
        model_labels = []
        
        for model_name in models:
            # Get the layer at the current position
            layer_name = list(model_activations[model_name].keys())[layer_index]
            print(f"Processing layer {layer_name} for model {model_name}")  # Debugging line
            
            activation = model_activations[model_name][layer_name]
            aggregated_activation = activation.mean(dim=1)  # Aggregate across tokens
            activations_by_model.append(aggregated_activation)
            model_labels.extend([model_name] * aggregated_activation.shape[0])
        
        # Concatenate activations across models for the current layer
        combined_activations = torch.cat(activations_by_model, dim=0)
        combined_rdm = compute_rdm(combined_activations)
        
        # Perform MDS on the combined RDM
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        points_2d = mds.fit_transform(combined_rdm)

        # Plot the combined RDM for the current layer
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10', len(models))
        
        # Plot each model's points with distinct colors
        start_idx = 0
        for i, model_name in enumerate(models):
            num_points = activations_by_model[i].shape[0]
            plt.scatter(points_2d[start_idx:start_idx + num_points, 0], 
                        points_2d[start_idx:start_idx + num_points, 1], 
                        s=100, color=colors(i), label=model_name, edgecolor="k")
            start_idx += num_points

        plt.title(f"Combined MDS Plot for Layer Position {layer_index} Across Models")
        plt.xlabel("MDS Dimension 1")
        plt.ylabel("MDS Dimension 2")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
