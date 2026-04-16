import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from huggingface_hub import HfApi


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    architecture: str
    size_label: str
    role: str
    model_type: str = "default"
    notes: str = ""


ANALYSIS_MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        model_id="answerdotai/ModernBERT-large",
        family="ModernBERT",
        architecture="encoder",
        size_label="395M",
        role="analysis",
        notes="Modern encoder baseline with stronger long-context support than legacy BERT models.",
    ),
    ModelSpec(
        model_id="microsoft/deberta-v3-large",
        family="DeBERTa-v3",
        architecture="encoder",
        size_label="435M",
        role="analysis",
        notes="Strong encoder with disentangled attention, useful contrast to BERT-style encoders.",
    ),
    ModelSpec(
        model_id="FacebookAI/roberta-large",
        family="RoBERTa",
        architecture="encoder",
        size_label="355M",
        role="analysis",
        notes="Continuity anchor to the older project while using a stronger checkpoint than roberta-base.",
    ),
    ModelSpec(
        model_id="FacebookAI/xlm-roberta-large",
        family="XLM-R",
        architecture="encoder",
        size_label="550M",
        role="analysis",
        notes="Multilingual encoder baseline; useful if the stimulus inventory later expands beyond English.",
    ),
    ModelSpec(
        model_id="Qwen/Qwen2.5-3B",
        family="Qwen2.5",
        architecture="decoder",
        size_label="3B",
        role="analysis",
        notes="Smaller modern decoder for scale comparisons within the same family.",
    ),
    ModelSpec(
        model_id="Qwen/Qwen2.5-7B",
        family="Qwen2.5",
        architecture="decoder",
        size_label="7B",
        role="analysis",
        notes="Modern decoder baseline with strong representation quality and open HF availability.",
    ),
    ModelSpec(
        model_id="mistralai/Mistral-Nemo-Base-2407",
        family="Mistral Nemo",
        architecture="decoder",
        size_label="12B",
        role="analysis",
        notes="Modern Mistral-family base model; preferable to older 7B v0.x checkpoints.",
    ),
    ModelSpec(
        model_id="allenai/OLMo-2-1124-7B",
        family="OLMo 2",
        architecture="decoder",
        size_label="7B",
        role="analysis",
        notes="Scientifically attractive open model with unusually transparent training details.",
    ),
]


GENERATION_MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        family="Qwen2.5",
        architecture="decoder",
        size_label="7B",
        role="generation",
        notes="Primary homonym-data generation model.",
    ),
    ModelSpec(
        model_id="mistralai/Mistral-Nemo-Instruct-2407",
        family="Mistral Nemo",
        architecture="decoder",
        size_label="12B",
        role="generation",
        notes="Fallback or comparison generator for synthetic data quality checks.",
    ),
]


SECONDARY_VALIDATION_MODEL_SPECS: List[ModelSpec] = list(ANALYSIS_MODEL_SPECS)


def get_analysis_model_specs() -> List[ModelSpec]:
    return list(ANALYSIS_MODEL_SPECS)


def get_generation_model_specs() -> List[ModelSpec]:
    return list(GENERATION_MODEL_SPECS)


def get_secondary_validation_model_specs() -> List[ModelSpec]:
    return list(SECONDARY_VALIDATION_MODEL_SPECS)


def get_model_config_dict(model_specs: Sequence[ModelSpec]) -> Dict[str, Dict[str, str]]:
    return {
        spec.model_id: {
            "model_type": spec.model_type,
            "family": spec.family,
            "architecture": spec.architecture,
            "size_label": spec.size_label,
            "role": spec.role,
        }
        for spec in model_specs
    }


def save_model_catalog(path: str, model_specs: Sequence[ModelSpec]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump([asdict(spec) for spec in model_specs], file, indent=2)


def verify_models_on_hf(
    model_specs: Sequence[ModelSpec],
    token: Optional[str] = None,
) -> List[Dict[str, object]]:
    api = HfApi(token=token)
    rows: List[Dict[str, object]] = []

    for spec in model_specs:
        try:
            info = api.model_info(spec.model_id)
            rows.append(
                {
                    "model_id": spec.model_id,
                    "available": True,
                    "sha": getattr(info, "sha", None),
                    "private": bool(getattr(info, "private", False)),
                    "gated": bool(getattr(info, "gated", False)),
                    "pipeline_tag": getattr(info, "pipeline_tag", None),
                }
            )
        except Exception as exc:  # pragma: no cover - depends on online HF access
            rows.append(
                {
                    "model_id": spec.model_id,
                    "available": False,
                    "error": str(exc),
                }
            )

    return rows
