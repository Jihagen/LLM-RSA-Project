import gc
import json
import logging
import os
from typing import Dict, List, Optional, Sequence

from configs import (
    DEFAULT_HPC_PIPELINE,
    PipelineConfig,
    get_analysis_model_specs,
    get_generation_model_specs,
    get_secondary_validation_model_specs,
    save_model_catalog,
    verify_models_on_hf,
)
from data import load_or_generate_homonym_dataset
from experiments import (
    run_across_model_profile_comparison,
    run_distributed_semantic_profile_experiment,
    run_layer_identification_experiment,
)
from models import load_model_and_tokenizer


def _load_pandas():
    import pandas as pd

    return pd


def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _select_model_specs(all_specs, allowed_model_ids: Sequence[str]):
    allowed = set(allowed_model_ids)
    return [spec for spec in all_specs if spec.model_id in allowed]


def _write_json(path: str, payload: Dict[str, object]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def _materialize_model_catalog(config: PipelineConfig) -> Dict[str, object]:
    analysis_specs = _select_model_specs(get_analysis_model_specs(), config.analysis_model_ids)
    generation_specs = _select_model_specs(get_generation_model_specs(), config.generation_model_ids)

    catalog_payload = {
        "analysis_models": [spec.__dict__ for spec in analysis_specs],
        "generation_models": [spec.__dict__ for spec in generation_specs],
    }
    save_model_catalog(config.outputs.model_catalog_path, analysis_specs + generation_specs)

    if config.verify_hf_catalog:
        availability_rows = verify_models_on_hf(analysis_specs + generation_specs)
        _write_json(
            config.outputs.hf_availability_report_path,
            {"rows": availability_rows},
        )
        catalog_payload["hf_availability_report_path"] = config.outputs.hf_availability_report_path

    return catalog_payload


def generate_or_load_dataset(config: PipelineConfig):
    pd = _load_pandas()
    generation_config = config.generation

    if config.generate_dataset:
        return load_or_generate_homonym_dataset(
            seed_inventory_path=generation_config.seed_inventory_path,
            generation_model_id=generation_config.generation_model_id,
            output_pickle_path=generation_config.output_pickle_path,
            output_jsonl_path=generation_config.output_jsonl_path,
            output_summary_path=generation_config.output_summary_path,
            examples_per_sense_target=generation_config.examples_per_sense_target,
            minimum_examples_per_sense=generation_config.minimum_examples_per_sense,
            request_size_per_generation_call=generation_config.request_size_per_generation_call,
            max_generation_rounds=generation_config.max_generation_rounds,
            random_seed=generation_config.random_seed,
            force_regenerate=False,
        )

    return pd.read_pickle(config.analysis.dataset_path)


def run_homonym_profile_suite(df, config: PipelineConfig) -> Dict[str, object]:
    analysis_specs = _select_model_specs(get_analysis_model_specs(), config.analysis_model_ids)
    experiment_results: Dict[str, Dict[str, object]] = {}

    for spec in analysis_specs:
        logging.info("Running distributed semantic profiles for %s", spec.model_id)
        try:
            result = run_distributed_semantic_profile_experiment(
                df=df,
                model_name=spec.model_id,
                model_type=spec.model_type,
                base_dir=config.analysis.base_dir,
                batch_size=config.analysis.batch_size,
                cv_splits=config.analysis.cv_splits,
                cv_repeats=config.analysis.cv_repeats,
                random_state=config.analysis.random_state,
            )
            experiment_results[spec.model_id] = result
        finally:
            gc.collect()

    cross_model_summary: Optional[Dict[str, object]] = None
    if config.analysis.run_cross_model_comparison and experiment_results:
        cross_model_summary = run_across_model_profile_comparison(
            experiment_results=experiment_results,
            base_dir=config.analysis.base_dir,
        )

    return {
        "experiment_results": experiment_results,
        "cross_model_summary": cross_model_summary,
    }


def run_wic_validation_suite(config: PipelineConfig) -> Dict[str, object]:
    validation_specs = _select_model_specs(
        get_secondary_validation_model_specs(),
        config.validation_model_ids,
    )
    results: List[Dict[str, str]] = []

    for spec in validation_specs:
        model = None
        tokenizer = None
        try:
            logging.info("Running secondary WiC validation for %s", spec.model_id)
            model, tokenizer = load_model_and_tokenizer(spec.model_id, model_type=spec.model_type)
            run_layer_identification_experiment(
                model=model,
                tokenizer=tokenizer,
                dataset_name="wic",
                split="train",
                results_dir=config.analysis.base_dir,
            )
            results.append({"model_id": spec.model_id, "status": "ok"})
        except Exception as exc:
            results.append({"model_id": spec.model_id, "status": f"failed: {exc}"})
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()

    return {"rows": results}


def run_full_pipeline(config: PipelineConfig = DEFAULT_HPC_PIPELINE) -> Dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model_catalog = _materialize_model_catalog(config)
    df = generate_or_load_dataset(config)

    summary: Dict[str, object] = {
        "model_catalog": model_catalog,
        "dataset_path": config.generation.output_pickle_path if config.generate_dataset else config.analysis.dataset_path,
        "run_homonym_profiles": config.run_homonym_profiles,
        "run_wic_validation": config.run_wic_validation,
    }

    if config.run_homonym_profiles:
        summary["homonym_profiles"] = run_homonym_profile_suite(df=df, config=config)

    if config.run_wic_validation:
        summary["wic_validation"] = run_wic_validation_suite(config=config)

    _write_json(config.outputs.pipeline_summary_path, summary)
    return summary
