import os
import torch
from transformers import AutoTokenizer, AutoModel

def load_model_and_tokenizer(model_name, model_type="default"):
    load_args = {}

    if model_type == "auth":
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token is None:
            raise ValueError("Authentication token required but HUGGINGFACE_HUB_TOKEN not set.")
        load_args["use_auth_token"] = token

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model in lower precision if possible
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, **load_args).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_args)

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_activations(model, tokenizer, texts, layer_indices=None, model_type="default"):
    """
    Returns a dictionary of activations from specified layers.
    """

    device = next(model.parameters()).device

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    )
    
    # Move tensors to the correct device and cast dtype for efficiency
    inputs = {key: value.to(device, dtype=torch.float16) for key, value in inputs.items()}

    activations = {}

    def hook_fn(idx):
        def hook(module, input, output):
            if hasattr(output, "last_hidden_state"):
                output_tensor = output.last_hidden_state
            elif isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            activations[idx] = output_tensor.detach().cpu()  # Store activations in CPU
        return hook

    if model_type == "encoder-decoder":
        for idx, (name, layer) in enumerate(model.encoder.named_modules()):
            if layer_indices is None or idx in layer_indices:
                layer.register_forward_hook(hook_fn(f"encoder_{idx}"))

        if "decoder_input_ids" not in inputs:
            decoder_input_ids = tokenizer(" ", return_tensors="pt").input_ids.to(device)
            inputs["decoder_input_ids"] = decoder_input_ids

    else:
        for idx, (name, layer) in enumerate(model.named_modules()):
            if layer_indices is None or idx in layer_indices:
                layer.register_forward_hook(hook_fn(idx))

    # Use no_grad() to reduce memory usage
    with torch.no_grad():
        model(**inputs)

    return activations
