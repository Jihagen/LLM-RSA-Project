import contextlib
import logging
from typing import Dict, List, Tuple

import torch
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from utils.hpc import build_hf_load_args, cleanup_torch, configure_hpc_runtime, model_device, should_use_cuda_autocast


logger = logging.getLogger(__name__)
configure_hpc_runtime()


_DECODER_ONLY_MODEL_TYPES = {
    'gpt2', 'gpt_neo', 'gpt_neox', 'gptj', 'bloom', 'opt',
    'llama', 'mistral', 'qwen2', 'olmo', 'falcon', 'gemma',
    'phi', 'stablelm', 'mpt', 'rwkv', 'codegen', 'xglm',
}


def is_decoder_only(model) -> bool:
    """Return True if model is a causal/decoder-only architecture."""
    cfg = model.config
    if getattr(cfg, 'is_decoder', False):
        return True
    archs = getattr(cfg, 'architectures', None) or []
    if any('causal' in a.lower() or 'generative' in a.lower() for a in archs):
        return True
    return getattr(cfg, 'model_type', '').lower() in _DECODER_ONLY_MODEL_TYPES


class TokenProbeModel(torch.nn.Module):
    """
    Wrapper that loads a pretrained Transformer and returns only the hidden states
    at the token positions corresponding to the homonym.
    """

    def __init__(self, model_name: str, model_type: str = "default"):
        super().__init__()
        source, load_args = _resolve_source_and_args(model_name, model_type)
        load_args["output_hidden_states"] = True
        self.model = _load_auto_model(source, load_args, model_name)
        self.model.eval()
        self.name_or_path = getattr(self.model, "name_or_path", model_name)

    @property
    def config(self):
        return self.model.config

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        homonym_positions: List[List[int]],
    ) -> Dict[int, List[torch.Tensor]]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        batch_size = input_ids.size(0)
        token_level_outputs: Dict[int, List[torch.Tensor]] = {}
        for layer_idx, layer_hs in enumerate(hidden_states):
            per_example_embeddings: List[torch.Tensor] = []
            for sample_idx in range(batch_size):
                positions = homonym_positions[sample_idx]
                if positions:
                    emb = layer_hs[sample_idx, positions, :]
                else:
                    emb = layer_hs.new_empty((0, layer_hs.size(-1)))
                per_example_embeddings.append(emb)
            token_level_outputs[layer_idx] = per_example_embeddings

        return token_level_outputs


def _resolve_source_and_args(model_name: str, model_type: str) -> Tuple[str, Dict[str, object]]:
    runtime = build_hf_load_args(model_name, model_type=model_type)
    source = runtime["source"]
    load_args = dict(runtime["load_args"])
    return source, load_args


def _load_auto_model(source: str, load_args: Dict[str, object], requested_name: str):
    try:
        return AutoModel.from_pretrained(source, **load_args)
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise ValueError(f"Model {requested_name} not found or inaccessible: {exc}") from exc
    except Exception as exc:
        if "custom code" in str(exc).lower() or "trust_remote_code" in str(exc).lower():
            retry_args = dict(load_args)
            retry_args["trust_remote_code"] = True
            logger.warning(
                "Custom code required for %s; retrying with trust_remote_code=True",
                requested_name,
            )
            return AutoModel.from_pretrained(source, **retry_args)
        raise


def _load_auto_tokenizer(source: str, load_args: Dict[str, object], requested_name: str):
    tokenizer_args = {
        key: value
        for key, value in load_args.items()
        if key in {"local_files_only", "token", "use_auth_token", "trust_remote_code"}
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_args)
    except Exception as exc:
        if "custom code" in str(exc).lower() or "trust_remote_code" in str(exc).lower():
            tokenizer_args["trust_remote_code"] = True
            logger.warning(
                "Custom tokenizer code required for %s; retrying with trust_remote_code=True",
                requested_name,
            )
            tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_args)
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tokenizer(model_name: str, model_type: str = "default"):
    source, load_args = _resolve_source_and_args(model_name, model_type)
    return _load_auto_tokenizer(source, load_args, model_name)


def load_model_and_tokenizer(model_name: str, model_type: str = "default"):
    source, load_args = _resolve_source_and_args(model_name, model_type)
    model = _load_auto_model(source, load_args, model_name)
    model.eval()
    tokenizer = _load_auto_tokenizer(source, load_args, model_name)
    return model, tokenizer


def _get_inference_context(device: torch.device):
    base = torch.inference_mode()
    if should_use_cuda_autocast() and device.type == "cuda":
        return contextlib.ExitStack().__enter__()
    return base


@contextlib.contextmanager
def _inference_context(device: torch.device):
    with torch.inference_mode():
        if should_use_cuda_autocast() and device.type == "cuda":
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            yield


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def get_activations(
    model,
    tokenizer,
    texts1,
    texts2,
    layer_indices=None,
    model_type="default",
    batch_size=16,
):
    """
    Returns a dictionary of activations for specified layers by processing text pairs in batches.
    For each sample, the activations are split into two parts using token_type_ids and
    mean-pooled over the tokens in each segment.
    """
    del model_type  # Kept for call-site compatibility.
    device = model_device(model)

    if layer_indices is None:
        layer_indices = list(range(model.config.num_hidden_layers + 1))
    all_activations = {layer_idx: ([], []) for layer_idx in layer_indices}

    num_samples = len(texts1)
    for start in range(0, num_samples, batch_size):
        batch_texts1 = texts1[start:start + batch_size]
        batch_texts2 = texts2[start:start + batch_size]
        inputs = tokenizer(
            batch_texts1,
            batch_texts2,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"].detach().cpu()
        else:
            input_ids = inputs["input_ids"].detach().cpu()
            sep_id = tokenizer.sep_token_id
            token_type_ids_list = []
            for sample in input_ids:
                sample_tokens = sample.tolist()
                try:
                    sep_index = sample_tokens.index(sep_id)
                except ValueError:
                    sep_index = len(sample_tokens)
                tt_ids = [0] * len(sample_tokens)
                for idx in range(sep_index + 1, len(sample_tokens)):
                    tt_ids[idx] = 1
                token_type_ids_list.append(torch.tensor(tt_ids, dtype=torch.long))
            token_type_ids = torch.stack(token_type_ids_list, dim=0)

        model_inputs = _move_batch_to_device(dict(inputs), device)

        with _inference_context(device):
            outputs = model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for idx in layer_indices:
            raw_act = hidden_states[idx].detach().to(torch.float32).cpu()
            bs = raw_act.size(0)
            activations_text1 = []
            activations_text2 = []
            for sample_idx in range(bs):
                mask1 = token_type_ids[sample_idx] == 0
                mask2 = token_type_ids[sample_idx] == 1
                pooled1 = raw_act[sample_idx][mask1].mean(dim=0) if mask1.any() else raw_act.new_zeros(raw_act.size(2))
                pooled2 = raw_act[sample_idx][mask2].mean(dim=0) if mask2.any() else raw_act.new_zeros(raw_act.size(2))
                activations_text1.append(pooled1.unsqueeze(0))
                activations_text2.append(pooled2.unsqueeze(0))

            all_activations[idx][0].append(torch.cat(activations_text1, dim=0))
            all_activations[idx][1].append(torch.cat(activations_text2, dim=0))

        cleanup_torch()

    final_activations = {}
    for idx, (list_text1, list_text2) in all_activations.items():
        act_text1 = torch.cat(list_text1, dim=0)
        act_text2 = torch.cat(list_text2, dim=0)
        final_activations[idx] = (act_text1, act_text2)

    return final_activations


def get_target_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    texts: List[str],
    targets: List[str],
    batch_size: int = 8,
    layer_indices: List[int] = None,
    pooling: str = "target",
) -> Dict[int, torch.Tensor]:
    """
    Extract per-sentence hidden-state vectors across all layers.

    pooling='target'     — mean-pool the subword tokens that cover the target word.
                           Correct for bidirectional encoders that see full context.
    pooling='last_token' — take the last non-padding token's hidden state.
                           Use this for causal/decoder-only models: the target word
                           may appear before the disambiguating context, so the last
                           token position captures the full available left context.
    """
    if pooling not in ("target", "last_token"):
        raise ValueError(f"pooling must be 'target' or 'last_token', got {pooling!r}")

    device = model_device(model)
    num_hidden = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = list(range(num_hidden + 1))

    all_acts = {layer: [] for layer in layer_indices}

    for start in range(0, len(texts), batch_size):
        batch_texts   = texts[start:start + batch_size]
        batch_targets = targets[start:start + batch_size]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offset_mappings = encoding.pop("offset_mapping")
        attention_mask  = encoding["attention_mask"]
        model_inputs    = _move_batch_to_device(dict(encoding), device)

        with _inference_context(device):
            outputs = model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        if pooling == "last_token":
            last_positions = (attention_mask.sum(dim=1) - 1).tolist()
        else:
            batch_masks: List[torch.Tensor] = []
            for offsets, target, text in zip(offset_mappings, batch_targets, batch_texts):
                offset_list  = offsets.tolist()
                lower_text   = text.lower()
                target_start = lower_text.find(target.lower())
                if target_start < 0:
                    batch_masks.append(torch.zeros(len(offset_list), dtype=torch.bool))
                    continue
                target_end = target_start + len(target)
                mask = [not (end <= target_start or begin >= target_end) for begin, end in offset_list]
                batch_masks.append(torch.tensor(mask, dtype=torch.bool))

        for layer in layer_indices:
            hidden = hidden_states[layer].detach().to(torch.float32).cpu()
            for sample_idx in range(hidden.size(0)):
                if pooling == "last_token":
                    pos = int(last_positions[sample_idx])
                    vec = hidden[sample_idx, pos : pos + 1, :]
                else:
                    mask = batch_masks[sample_idx]
                    if mask.any():
                        vec = hidden[sample_idx][mask].mean(dim=0, keepdim=True)
                    else:
                        vec = torch.zeros((1, hidden.size(-1)), dtype=torch.float32)
                all_acts[layer].append(vec)

        cleanup_torch()

    return {layer: torch.cat(all_acts[layer], dim=0) for layer in layer_indices}
