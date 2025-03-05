import os
import torch
from transformers import AutoTokenizer, AutoModel
from torch import cuda
from torch.amp import autocast
import torch.nn.functional as F

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

def pad_and_cat(tensor_list, dim=0, pad_value=0.0):
    """
    Pads each tensor in tensor_list along dimension 1 (sequence length)
    to the maximum size found, then concatenates along dimension `dim`.
    """
    max_seq_len = max(t.shape[1] for t in tensor_list)
    padded_tensors = []
    for t in tensor_list:
        seq_len = t.shape[1]
        if seq_len < max_seq_len:
            pad_amount = max_seq_len - seq_len
            t_padded = F.pad(t, (0, 0, 0, pad_amount), "constant", pad_value)
            padded_tensors.append(t_padded)
        else:
            padded_tensors.append(t)
    return torch.cat(padded_tensors, dim=dim)

def get_activations(model, tokenizer, texts, layer_indices=None, model_type="default", batch_size=8):
    """
    Returns a dictionary of activations from specified layers by processing texts in batches.
    Dynamically pads activations so that tensors from different batches can be concatenated.
    """
    device = next(model.parameters()).device
    all_activations = {}  # Dictionary: layer index -> list of tensors from each batch

    num_texts = len(texts)
    for start in range(0, num_texts, batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        # Move inputs to the device (keep input_ids as ints)
        for key, value in inputs.items():
            if key in ["input_ids", "token_type_ids"]:
                inputs[key] = value.to(device)
            else:
                inputs[key] = value.to(device, dtype=torch.float16)
        
        batch_activations = {}  # Store activations for this batch
        hook_handles = []

        def hook_fn(idx):
            def hook(module, input, output):
                if hasattr(output, "last_hidden_state"):
                    output_tensor = output.last_hidden_state
                elif isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                # Save activations on CPU
                batch_activations.setdefault(idx, []).append(output_tensor.detach().cpu())
            return hook

        if model_type == "encoder-decoder":
            for idx, (name, layer) in enumerate(model.encoder.named_modules()):
                if layer_indices is None or idx in layer_indices:
                    handle = layer.register_forward_hook(hook_fn(f"encoder_{idx}"))
                    hook_handles.append(handle)
            if "decoder_input_ids" not in inputs:
                decoder_input_ids = tokenizer(" ", return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                inputs["decoder_input_ids"] = decoder_input_ids
        else:
            for idx, (name, layer) in enumerate(model.named_modules()):
                if layer_indices is None or idx in layer_indices:
                    handle = layer.register_forward_hook(hook_fn(idx))
                    hook_handles.append(handle)

        with torch.no_grad(), autocast(device_type='cuda'):
            model(**inputs)

        for handle in hook_handles:
            handle.remove()
        torch.cuda.empty_cache()
        
        for idx, acts_list in batch_activations.items():
            if idx in all_activations:
                all_activations[idx].extend(acts_list)
            else:
                all_activations[idx] = acts_list.copy()
        
        import gc
        gc.collect()

    for idx in all_activations:
        all_activations[idx] = pad_and_cat(all_activations[idx], dim=0)
    
    return all_activations
