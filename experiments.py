# Main functions for each experiment

from activations import load_model_and_tokenizer, get_activations, compute_rdm
from visualisation import plot_rdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE

def compare_activations_per_layer(models, texts):
    for model_name in models:
        model, tokenizer = load_model_and_tokenizer(model_name)
        all_activations = get_activations(model, tokenizer, texts)

        for layer, activation in all_activations.items():
            rdm = compute_rdm(activation)
            plot_rdm(rdm, f"RDM for {model_name} - {layer}", method="MDS")

def compare_activations_per_sample(models, samples):
    for sample in samples:
        activations_by_model = []
        model_labels = []
        
        # Gather activations for each model
        for model_name in models:
            model, tokenizer = load_model_and_tokenizer(model_name)
            activations = get_activations(model, tokenizer, [sample])
            first_layer_activation = activations[next(iter(activations))]
            
            # Append activations and labels
            activations_by_model.append(first_layer_activation)
            model_labels.extend([model_name] * first_layer_activation.shape[0])
        
        # Concatenate activations across models
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

def compare_activations_per_layer_modelSpec(model_name, texts):
    model, tokenizer = load_model_and_tokenizer(model_name)
    all_activations = get_activations(model, tokenizer, texts)

    for layer, activation in all_activations.items():
        rdm = compute_rdm(activation)
        plot_rdm(rdm, f"Layer {layer} Across Samples", method="MDS")
