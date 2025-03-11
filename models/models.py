import os
import torch
from transformers import AutoTokenizer, AutoModel
from torch import cuda
from torch.amp import autocast
import torch.nn.functional as F
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

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

    try:
        # Try loading the model normally
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, **load_args).to(device)
        model.eval()
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        raise ValueError(f"Model {model_name} not found or inaccessible: {e}")
    except Exception as e:
        if "custom code" in str(e).lower():  # Check if error message suggests remote code execution is needed
            print(f"⚠️ Detected custom code requirement for {model_name}. Retrying with trust_remote_code=True...")
            load_args["trust_remote_code"] = True
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, **load_args).to(device)
            model.eval()
        else:
            raise e  # Raise the error if it's unrelated to trust_remote_code
    # Try loading the tokenizer normally
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_args)
    except Exception as e:
        if "custom code" in str(e).lower():
            print(f"⚠️ Detected custom code requirement for {model_name} tokenizer. Retrying with trust_remote_code=True...")
            load_args["trust_remote_code"] = True
            tokenizer = AutoTokenizer.from_pretrained(model_name, **load_args)
        else:
            raise e  # Raise the error if it's unrelated to trust_remote_code

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
    

def get_activations(model, tokenizer, texts1, texts2, layer_indices=None, model_type="default", batch_size=16):
    """
    Returns a dictionary of activations for specified layers by processing text pairs in batches.
    For each sample, the activations are split into two parts (one per text in the pair) using token_type_ids,
    and mean pooling is applied over the tokens belonging to each segment.
    
    Args:
        model: Pre-trained transformer model.
        tokenizer: Hugging Face tokenizer.
        texts1: List of strings for the first sentence.
        texts2: List of strings for the second sentence.
        layer_indices: Optional list of layer indices to hook.
        model_type: (unused here) Model type specifier.
        batch_size: Batch size for processing.
    
    Returns:
        final_activations: A dictionary mapping each layer index to a tuple (act_text1, act_text2), where each
                           is a tensor of shape [N, hidden_dim] (N = total number of samples).
    """
    device = next(model.parameters()).device
    # For each layer index, we will accumulate two lists: one for activations from text1, one for text2.
    # We initialize the dictionary for each requested layer.
    if layer_indices is None:
        layer_indices = list(range(model.config.num_hidden_layers))
    all_activations = {layer_idx: ([], []) for layer_idx in layer_indices}

    num_samples = len(texts1)
    for start in range(0, num_samples, batch_size):
        batch_texts1 = texts1[start:start+batch_size]
        batch_texts2 = texts2[start:start+batch_size]
        # Tokenize the text pair.
        inputs = tokenizer(batch_texts1, batch_texts2, return_tensors="pt", padding=True, truncation=True)
        # Save token_type_ids from inputs (shape: [batch_size, seq_len]); these indicate segment membership.
        token_type_ids = inputs["token_type_ids"].detach().cpu()
        # Move inputs to device.
        for key, value in inputs.items():
            if key in ["input_ids", "token_type_ids"]:
                inputs[key] = value.to(device)
            else:
                inputs[key] = value.to(device, dtype=torch.float16)
        
        # Prepare a temporary dictionary for raw activations in this batch.
        batch_activations = {}  # Maps layer index -> tensor of shape [batch_size, seq_len, hidden_dim]
        hook_handles = []
        
        def hook_fn(idx):
            def hook(module, input, output):
                if hasattr(output, "last_hidden_state"):
                    output_tensor = output.last_hidden_state
                elif isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                batch_activations[idx] = output_tensor.detach().cpu()
            return hook
        
        # Register hooks for the layers we want.
        # (This example uses model.named_modules()—adjust as needed for your model structure.)
        for name, layer in model.named_modules():
            try:
                # Try to interpret the module's name as a layer index.
                idx = int(name)
            except (ValueError, TypeError):
                continue  # Skip modules that don’t have a numeric name.
            if idx in layer_indices:
                handle = layer.register_forward_hook(hook_fn(idx))
                hook_handles.append(handle)
        
        # Run the model.
        with torch.no_grad(), autocast(device_type='cuda'):
            model(**inputs)
        # Remove hooks.
        for handle in hook_handles:
            handle.remove()
        torch.cuda.empty_cache()
        
        # Process the raw activations for each hooked layer.
        # For each layer, we split the activations into two parts using token_type_ids.
        for idx, raw_act in batch_activations.items():
            # raw_act shape: [batch_size, seq_len, hidden_dim]
            bs = raw_act.size(0)
            activations_text1 = []
            activations_text2 = []
            for i in range(bs):
                # Create masks for tokens belonging to the first and second segments.
                mask1 = token_type_ids[i] == 0
                mask2 = token_type_ids[i] == 1
                # Mean pool over tokens in each segment. If no token is found, create a zero vector.
                if mask1.sum() > 0:
                    pooled1 = raw_act[i][mask1].mean(dim=0)
                else:
                    pooled1 = raw_act[i].new_zeros(raw_act.size(2))
                if mask2.sum() > 0:
                    pooled2 = raw_act[i][mask2].mean(dim=0)
                else:
                    pooled2 = raw_act[i].new_zeros(raw_act.size(2))
                activations_text1.append(pooled1.unsqueeze(0))
                activations_text2.append(pooled2.unsqueeze(0))
            # Stack along the batch dimension.
            activations_text1 = torch.cat(activations_text1, dim=0)  # [batch_size, hidden_dim]
            activations_text2 = torch.cat(activations_text2, dim=0)  # [batch_size, hidden_dim]
            # Append these batch results to the overall lists.
            all_activations[idx][0].append(activations_text1)
            all_activations[idx][1].append(activations_text2)
        
        import gc
        gc.collect()

    # Concatenate batches for each layer.
    final_activations = {}
    for idx, (list_text1, list_text2) in all_activations.items():
        act_text1 = torch.cat(list_text1, dim=0)
        act_text2 = torch.cat(list_text2, dim=0)
        final_activations[idx] = (act_text1, act_text2)
    
    return final_activations
