import os
import torch
from transformers import AutoTokenizer, AutoModel
from torch import cuda
from torch.amp import autocast
import torch.nn.functional as F
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from data import * 
from typing import List, Tuple, Dict
import inspect


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
        
        # Check if token_type_ids are provided; if not, generate them manually.
        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"].detach().cpu()
        else:
            # Manually create token_type_ids for text pairs.
            input_ids = inputs["input_ids"].detach().cpu()
            sep_id = tokenizer.sep_token_id
            token_type_ids_list = []
            for sample in input_ids:
                sample = sample.tolist()
                try:
                    # Find the first occurrence of the separator token.
                    sep_index = sample.index(sep_id)
                except ValueError:
                    sep_index = len(sample)
                # Tokens before (and including) the separator get 0; the rest get 1.
                tt_ids = [0] * len(sample)
                for i in range(sep_index + 1, len(sample)):
                    tt_ids[i] = 1
                token_type_ids_list.append(torch.tensor(tt_ids, dtype=torch.long))
            token_type_ids = torch.stack(token_type_ids_list, dim=0)
        
        # Move inputs to the device.
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
                print(f"Layer {idx}: Activation shape {output_tensor.shape}")  # Debugging output
                batch_activations[idx] = output_tensor.detach().cpu()
            return hook

        # Register hooks for the layers we want.
        for idx, (name, layer) in enumerate(model.named_modules()):
            # You may wish to refine this selection logic based on your model architecture.
            if layer_indices is None or idx in layer_indices:
                print(f"Hooking layer {idx}: {name}")  # Debugging output
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
   
        print(f"Layer {idx}: Collected {len(list_text1)} batches for text1, {len(list_text2)} batches for text2")
        if len(list_text1) > 0:
            print(f"Example shape: {list_text1[0].shape}")

    return final_activations

from typing import List, Dict
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

def get_target_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    texts: List[str],
    targets: List[str],
    batch_size: int = 8,
    layer_indices: List[int] = None
) -> Dict[int, torch.Tensor]:
    """
    Tokenizes each batch ONCE with return_offsets_mapping=True,
    then for each sample uses the same offsets to find the target
    tokens, pools those positions in every hidden_state layer,
    and finally concatenates.
    """
    device = next(model.parameters()).device

    # embeddings + all hidden layers
    num_hidden = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = list(range(num_hidden + 1))

    all_acts = {l: [] for l in layer_indices}

    for start in range(0, len(texts), batch_size):
        batch_texts   = texts[start:start+batch_size]
        batch_targets = targets[start:start+batch_size]

        # === single tokenize call ===
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,   # keep the special tokens so offsets align
        ).to(device)

        offset_mappings = encoding.pop("offset_mapping")  # [B, T, 2]

        # forward
        with torch.no_grad():
            outputs = model(**encoding, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple: (emb, layer1, layer2, ...)

       
        batch_masks: List[torch.Tensor] = []
        for om, target, txt in zip(offset_mappings, batch_targets, batch_texts):
            om = om.tolist()
            lower = txt.lower()
            idx = lower.find(target.lower())
            if idx < 0:
                batch_masks.append(torch.zeros(len(om), dtype=torch.bool))
                continue
            start_char, end_char = idx, idx + len(target)
            # intersection test:
            mask = [ not (e <= start_char or s >= end_char) for (s,e) in om ]
            batch_masks.append(torch.tensor(mask, dtype=torch.bool))

        # now pool per‐layer
        for layer in layer_indices:
            h = hidden_states[layer].cpu()   # [B, T, H]
            for i, mask in enumerate(batch_masks):
                if mask.any():
                    # mean over the True positions
                    vec = h[i][mask].mean(dim=0, keepdim=True)
                else:
                    # fallback zero
                    vec = torch.zeros((1, h.size(-1)))
                all_acts[layer].append(vec)

    # concat each layer’s list → [N, H]
    return {l: torch.cat(all_acts[l], dim=0) for l in layer_indices}

