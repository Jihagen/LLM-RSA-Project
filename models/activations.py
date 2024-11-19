 # Functions for tokenizing, getting activations
import torch
from transformers import AutoTokenizer, AutoModel

def load_model_and_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_activations(model, tokenizer, texts, layer_indices=None):
    """
    Extract activations from specific layers of a model.

    Args:
        model: Hugging Face model from which to extract activations.
        tokenizer: Hugging Face tokenizer for text preprocessing.
        texts (list of str): Input texts for the model.
        layer_indices (list of int, optional): Indices of layers to extract activations from.
            If None, extract activations from all layers.

    Returns:
        dict: A dictionary where keys are layer indices and values are activation tensors.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    activations = {}

    def hook_fn(idx):
        def hook(model, input, output):
            if hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            elif isinstance(output, tuple):
                output = output[0]
            activations[idx] = output.detach()
        return hook

    # Register hooks only for specified layers
    for idx, (name, layer) in enumerate(model.named_modules()):
        if layer_indices is None or idx in layer_indices:
            layer.register_forward_hook(hook_fn(idx))
    
    # Perform a forward pass to collect activations
    with torch.no_grad():
        model(**inputs)
    
    return activations

