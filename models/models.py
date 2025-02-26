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


import torch
from torch import cuda
from torch.amp import autocast

def get_activations(model, tokenizer, texts, layer_indices=None, model_type="default", batch_size=8):
    """
    Returns a dictionary of activations from specified layers by processing texts in batches.
    """
    device = next(model.parameters()).device
    all_activations = {}  # Will store concatenated activations for each layer index

    # We'll process texts in batches
    num_texts = len(texts)
    for start in range(0, num_texts, batch_size):
        batch_texts = texts[start:start + batch_size]
        # Tokenize the batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        # For keys representing indices, don't cast to float16.
        for key, value in inputs.items():
            if key in ["input_ids", "token_type_ids"]:
                inputs[key] = value.to(device)
            else:
                inputs[key] = value.to(device, dtype=torch.float16)
        
        batch_activations = {}
        hook_handles = []

        def hook_fn(idx):
            def hook(module, input, output):
                if hasattr(output, "last_hidden_state"):
                    output_tensor = output.last_hidden_state
                elif isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                # Store activations from this batch on CPU
                batch_activations.setdefault(idx, []).append(output_tensor.detach().cpu())
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

        with torch.no_grad(), autocast(device_type='cuda'):
            model(**inputs)

        # Remove hooks and clear cache after the batch
        for handle in hook_handles:
            handle.remove()
        torch.cuda.empty_cache()

        # Concatenate activations for this batch into the overall dictionary
        for idx, acts_list in batch_activations.items():
            batch_concat = torch.cat(acts_list, dim=0)
            if idx in all_activations:
                all_activations[idx] = torch.cat([all_activations[idx], batch_concat], dim=0)
            else:
                all_activations[idx] = batch_concat

        # Optionally force garbage collection
        import gc
        gc.collect()

    return all_activations

