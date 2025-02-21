 # Functions for tokenizing, getting activations
import torch
from transformers import AutoTokenizer, AutoModel

def load_model_and_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer
    
def get_activations(model, tokenizer, texts, layer_indices=None):
    # Tokenize inputs
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Get device from model parameters
    device = next(model.parameters()).device
    # Move all input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    activations = {}

    def hook_fn(idx):
        def hook(model, input, output):
            if hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            elif isinstance(output, tuple):
                output = output[0]
            # Detach and move to CPU (optional) to free GPU memory after extraction
            activations[idx] = output.detach().cpu()
        return hook

    for idx, (name, layer) in enumerate(model.named_modules()):
        if layer_indices is None or idx in layer_indices:
            layer.register_forward_hook(hook_fn(idx))
    
    with torch.no_grad():
        model(**inputs)
    
    return activations
