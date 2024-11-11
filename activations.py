 # Functions for tokenizing, getting activations, and computing RDMs


import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import pdist, squareform

def load_model_and_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_activations(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    activations = {}

    def hook_fn(name):
        def hook(model, input, output):
            if hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            elif isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook

    for name, layer in model.named_modules():
        layer.register_forward_hook(hook_fn(name))
    
    with torch.no_grad():
        model(**inputs)
    
    return activations

def compute_rdm(activation_tensor, metric="cosine"):
    act_flat = activation_tensor.view(activation_tensor.size(0), -1).cpu().numpy()
    distances = pdist(act_flat, metric=metric)
    return squareform(distances)
