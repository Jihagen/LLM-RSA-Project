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

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_args)

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
def get_activations(model, tokenizer, texts, layer_indices=None, model_type="default"):
    device = next(model.parameters()).device
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # For keys representing indices, don't cast to float16.
    for key, value in inputs.items():
        if key in ["input_ids", "token_type_ids"]:
            inputs[key] = value.to(device)
        else:
            inputs[key] = value.to(device, dtype=torch.float16)
    
    activations = {}
    hook_handles = []  # Store hook handles so we can remove them

    def hook_fn(idx):
        def hook(module, input, output):
            if hasattr(output, "last_hidden_state"):
                output_tensor = output.last_hidden_state
            elif isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            activations[idx] = output_tensor.detach().cpu()
        return hook

    if model_type == "encoder-decoder":
        for idx, (name, layer) in enumerate(model.encoder.named_modules()):
            if layer_indices is None or idx in layer_indices:
                handle = layer.register_forward_hook(hook_fn(f"encoder_{idx}"))
                hook_handles.append(handle)
        if "decoder_input_ids" not in inputs:
            decoder_input_ids = tokenizer(" ", return_tensors="pt").input_ids.to(device)
            inputs["decoder_input_ids"] = decoder_input_ids
    else:
        for idx, (name, layer) in enumerate(model.named_modules()):
            if layer_indices is None or idx in layer_indices:
                handle = layer.register_forward_hook(hook_fn(idx))
                hook_handles.append(handle)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        model(**inputs)
    
    # Remove hooks after use to free resources
    for handle in hook_handles:
        handle.remove()
    
    # Clear cached GPU memory
    torch.cuda.empty_cache()
    
    return activations

