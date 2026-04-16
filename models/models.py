import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class ActivationCollection:
    representations: Dict[str, Dict[int, torch.Tensor]]
    metadata: Dict[str, List[object]]


class TokenProbeModel(torch.nn.Module):
    """
    Legacy wrapper kept for compatibility with older scripts.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.name_or_path = model_name
        self.config = self.model.config

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        homonym_positions: list[list[int]],
    ) -> dict[int, list[torch.Tensor]]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        token_level_outputs: dict[int, list[torch.Tensor]] = {}
        for layer_idx, layer_hs in enumerate(hidden_states):
            per_example_embeddings: list[torch.Tensor] = []
            for sample_index, positions in enumerate(homonym_positions):
                if positions:
                    embedding = layer_hs[sample_index, positions, :]
                else:
                    embedding = layer_hs.new_empty((0, layer_hs.size(-1)))
                per_example_embeddings.append(embedding)
            token_level_outputs[layer_idx] = per_example_embeddings

        return token_level_outputs

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)


def load_model_and_tokenizer(model_name, model_type="default"):
    load_args = {}

    if model_type == "auth":
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token is None:
            raise ValueError("Authentication token required but HUGGINGFACE_HUB_TOKEN not set.")
        load_args["use_auth_token"] = token

    if torch.cuda.is_available():
        load_args["low_cpu_mem_usage"] = True

    device_map = os.environ.get("HF_DEVICE_MAP")
    if device_map:
        load_args["device_map"] = device_map

    if os.environ.get("HF_TRUST_REMOTE_CODE", "0") == "1":
        load_args["trust_remote_code"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, **load_args)
        if "device_map" not in load_args:
            model = model.to(device)
        model.eval()
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise ValueError(f"Model {model_name} not found or inaccessible: {exc}") from exc
    except Exception as exc:
        if "custom code" in str(exc).lower():
            load_args["trust_remote_code"] = True
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, **load_args)
            if "device_map" not in load_args:
                model = model.to(device)
            model.eval()
        else:
            raise

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_args)
    except Exception as exc:
        if "custom code" in str(exc).lower():
            load_args["trust_remote_code"] = True
            tokenizer = AutoTokenizer.from_pretrained(model_name, **load_args)
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def find_first_target_span(text: str, target: str) -> Optional[Tuple[int, int]]:
    if not text or not target:
        return None

    escaped = re.escape(target)
    boundary_pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
    match = boundary_pattern.search(text)
    if match:
        return match.span()

    lowered_text = text.lower()
    lowered_target = target.lower()
    index = lowered_text.find(lowered_target)
    if index < 0:
        return None
    return index, index + len(target)


def _special_token_mask(offset_mapping: torch.Tensor) -> torch.Tensor:
    return (offset_mapping[:, 0] == 0) & (offset_mapping[:, 1] == 0)


def _build_masks(
    offset_mapping: torch.Tensor,
    attention_mask: torch.Tensor,
    target_span: Optional[Tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = attention_mask.bool()
    special_mask = _special_token_mask(offset_mapping)
    content_mask = valid_mask & ~special_mask

    if target_span is None:
        target_mask = torch.zeros_like(content_mask)
        return target_mask, content_mask

    start_char, end_char = target_span
    start_offsets = offset_mapping[:, 0]
    end_offsets = offset_mapping[:, 1]
    overlap_mask = (end_offsets > start_char) & (start_offsets < end_char)
    target_mask = content_mask & overlap_mask
    return target_mask, content_mask


def _pool_masked_hidden_state(
    hidden_state: torch.Tensor,
    mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if not mask.any():
        return hidden_state.new_zeros((hidden_state.size(-1),))

    masked_hidden = hidden_state[mask]
    if pooling == "mean":
        return masked_hidden.mean(dim=0)
    if pooling == "last":
        return masked_hidden[-1]
    raise ValueError(f"Unsupported pooling strategy: {pooling}")


def collect_target_span_representations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 8,
    layer_indices: Optional[Sequence[int]] = None,
    representation_kinds: Sequence[str] = ("target_mean", "target_last", "sentence_mean"),
) -> ActivationCollection:
    if len(texts) != len(targets):
        raise ValueError("texts and targets must have the same length.")

    device = next(model.parameters()).device
    num_hidden_layers = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = list(range(num_hidden_layers + 1))
    else:
        layer_indices = list(layer_indices)

    available_representations = {
        "target_mean": "mean",
        "target_last": "last",
        "sentence_mean": "mean",
    }
    unsupported = [name for name in representation_kinds if name not in available_representations]
    if unsupported:
        raise ValueError(f"Unsupported representation kinds requested: {unsupported}")

    all_representations: Dict[str, Dict[int, List[torch.Tensor]]] = {
        rep_name: {layer: [] for layer in layer_indices}
        for rep_name in representation_kinds
    }
    metadata: Dict[str, List[object]] = {
        "sample_index": [],
        "text": [],
        "target": [],
        "target_span": [],
        "target_found": [],
    }

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = list(texts[batch_start : batch_start + batch_size])
        batch_targets = list(targets[batch_start : batch_start + batch_size])

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offset_mapping = encoding.pop("offset_mapping")
        attention_mask = encoding["attention_mask"]
        model_inputs = {key: value.to(device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        for sample_offset, (text, target) in enumerate(zip(batch_texts, batch_targets)):
            sample_index = batch_start + sample_offset
            span = find_first_target_span(text, target)
            target_mask, content_mask = _build_masks(
                offset_mapping=offset_mapping[sample_offset],
                attention_mask=attention_mask[sample_offset],
                target_span=span,
            )

            metadata["sample_index"].append(sample_index)
            metadata["text"].append(text)
            metadata["target"].append(target)
            metadata["target_span"].append(span)
            metadata["target_found"].append(bool(target_mask.any()))

            for layer in layer_indices:
                layer_hidden = hidden_states[layer][sample_offset].detach().cpu()
                if "target_mean" in representation_kinds:
                    all_representations["target_mean"][layer].append(
                        _pool_masked_hidden_state(layer_hidden, target_mask, "mean")
                    )
                if "target_last" in representation_kinds:
                    all_representations["target_last"][layer].append(
                        _pool_masked_hidden_state(layer_hidden, target_mask, "last")
                    )
                if "sentence_mean" in representation_kinds:
                    all_representations["sentence_mean"][layer].append(
                        _pool_masked_hidden_state(layer_hidden, content_mask, "mean")
                    )

    stacked_representations: Dict[str, Dict[int, torch.Tensor]] = {}
    for rep_name, layer_dict in all_representations.items():
        stacked_representations[rep_name] = {
            layer: torch.stack(vectors, dim=0) if vectors else torch.empty((0, 0))
            for layer, vectors in layer_dict.items()
        }

    return ActivationCollection(representations=stacked_representations, metadata=metadata)


def get_target_activations(
    model: PreTrainedModel,
    tokenizer,
    texts: List[str],
    targets: List[str],
    batch_size: int = 8,
    layer_indices: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    collection = collect_target_span_representations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        targets=targets,
        batch_size=batch_size,
        layer_indices=layer_indices,
        representation_kinds=("target_mean",),
    )
    return collection.representations["target_mean"]


def collect_text_pair_representations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts1: Sequence[str],
    texts2: Sequence[str],
    batch_size: int = 8,
    layer_indices: Optional[Sequence[int]] = None,
) -> Dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if len(texts1) != len(texts2):
        raise ValueError("texts1 and texts2 must have the same length.")

    device = next(model.parameters()).device
    num_hidden_layers = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = list(range(num_hidden_layers + 1))
    else:
        layer_indices = list(layer_indices)

    collected: Dict[int, tuple[List[torch.Tensor], List[torch.Tensor]]] = {
        layer: ([], [])
        for layer in layer_indices
    }

    for batch_start in range(0, len(texts1), batch_size):
        batch_texts1 = list(texts1[batch_start : batch_start + batch_size])
        batch_texts2 = list(texts2[batch_start : batch_start + batch_size])
        encoding = tokenizer(
            batch_texts1,
            batch_texts2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping")
        attention_mask = encoding["attention_mask"]
        if getattr(encoding, "encodings", None):
            sequence_id_lists = [encoding.encodings[i].sequence_ids for i in range(len(batch_texts1))]
        elif "token_type_ids" in encoding:
            sequence_id_lists = [encoding["token_type_ids"][i].tolist() for i in range(len(batch_texts1))]
        else:
            sep_id = tokenizer.sep_token_id
            sequence_id_lists = []
            for input_ids in encoding["input_ids"]:
                input_id_list = input_ids.tolist()
                if sep_id is None or sep_id not in input_id_list:
                    sequence_id_lists.append([0] * len(input_id_list))
                    continue
                first_sep = input_id_list.index(sep_id)
                seq_ids = [0] * len(input_id_list)
                for pos in range(first_sep + 1, len(input_id_list)):
                    seq_ids[pos] = 1
                sequence_id_lists.append(seq_ids)
        model_inputs = {key: value.to(device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)

        for sample_offset, sequence_ids in enumerate(sequence_id_lists):
            sequence_ids_tensor = torch.tensor(
                [sid if sid is not None else -1 for sid in sequence_ids],
                dtype=torch.long,
            )
            valid_mask = attention_mask[sample_offset].bool()
            special_mask = _special_token_mask(offset_mapping[sample_offset])
            text1_mask = valid_mask & ~special_mask & (sequence_ids_tensor == 0)
            text2_mask = valid_mask & ~special_mask & (sequence_ids_tensor == 1)

            for layer in layer_indices:
                layer_hidden = outputs.hidden_states[layer][sample_offset].detach().cpu()
                pooled1 = _pool_masked_hidden_state(layer_hidden, text1_mask, "mean")
                pooled2 = _pool_masked_hidden_state(layer_hidden, text2_mask, "mean")
                collected[layer][0].append(pooled1)
                collected[layer][1].append(pooled2)

    return {
        layer: (
            torch.stack(text1_vectors, dim=0),
            torch.stack(text2_vectors, dim=0),
        )
        for layer, (text1_vectors, text2_vectors) in collected.items()
    }


def get_activations(
    model,
    tokenizer,
    texts1,
    texts2,
    layer_indices=None,
    model_type="default",
    batch_size=16,
):
    return collect_text_pair_representations(
        model=model,
        tokenizer=tokenizer,
        texts1=texts1,
        texts2=texts2,
        batch_size=batch_size,
        layer_indices=layer_indices,
    )
