from dataclasses import dataclass, field
from typing import List

from .model_registry import (
    get_analysis_model_specs,
    get_generation_model_specs,
    get_secondary_validation_model_specs,
)


@dataclass(frozen=True)
class GenerationConfig:
    seed_inventory_path: str = "data/homonym_seed_inventory.json"
    output_pickle_path: str = "data/generated/homonym_dataset.pkl"
    output_jsonl_path: str = "data/generated/homonym_dataset.jsonl"
    output_summary_path: str = "data/generated/homonym_dataset_summary.json"
    generation_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    examples_per_sense_target: int = 100
    minimum_examples_per_sense: int = 60
    request_size_per_generation_call: int = 20
    max_generation_rounds: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 1200
    random_seed: int = 42


@dataclass(frozen=True)
class AnalysisConfig:
    dataset_path: str = "data/generated/homonym_dataset.pkl"
    base_dir: str = "results"
    batch_size: int = 4
    cv_splits: int = 5
    cv_repeats: int = 3
    random_state: int = 42
    run_cross_model_comparison: bool = True


@dataclass(frozen=True)
class PipelineOutputConfig:
    manifest_dir: str = "results/manifests"
    model_catalog_path: str = "results/manifests/model_catalog.json"
    hf_availability_report_path: str = "results/manifests/hf_availability.json"
    pipeline_summary_path: str = "results/manifests/pipeline_summary.json"


@dataclass(frozen=True)
class PipelineConfig:
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    outputs: PipelineOutputConfig = field(default_factory=PipelineOutputConfig)
    analysis_model_ids: List[str] = field(default_factory=list)
    validation_model_ids: List[str] = field(default_factory=list)
    generation_model_ids: List[str] = field(default_factory=list)
    generate_dataset: bool = True
    run_homonym_profiles: bool = True
    run_wic_validation: bool = False
    verify_hf_catalog: bool = False


GENERATION_PRESET = GenerationConfig()
ANALYSIS_PRESET = AnalysisConfig()
PIPELINE_OUTPUT_PRESET = PipelineOutputConfig()
DEFAULT_HPC_PIPELINE = PipelineConfig(
    generation=GENERATION_PRESET,
    analysis=ANALYSIS_PRESET,
    outputs=PIPELINE_OUTPUT_PRESET,
    analysis_model_ids=[spec.model_id for spec in get_analysis_model_specs()],
    validation_model_ids=[spec.model_id for spec in get_secondary_validation_model_specs()],
    generation_model_ids=[spec.model_id for spec in get_generation_model_specs()],
    generate_dataset=True,
    run_homonym_profiles=True,
    run_wic_validation=False,
    verify_hf_catalog=False,
)
