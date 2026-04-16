# HPC Handover

## Project State

This repository was reworked into a homonym-first analysis pipeline for studying distributed semantic representations in language models.

The previous project state mixed several ideas:

- WiC-based layer identification
- homonym-based comparisons
- GDV-only analyses
- partially buggy activation aggregation

The current rebuild treats the homonym project as the primary scientific path and treats WiC only as a secondary validation dataset.

## Scientific Reframing

The main conceptual change is this:

- We no longer assume that each model has one single global "semantics layer".
- We now assume that semantic information may peak at different layers for different homonyms or senses.
- The primary question is therefore whether concept-wise layer profiles are structured, stable, and comparable within and across models.

This means:

- H1 is now about where each homonym sense contrast is best represented within a model.
- H2 is now about whether different models show corresponding homonym-specific semantic trajectories, not necessarily identical best layers.

## What Was Implemented

### 1. Activation Collection

The activation logic was reworked so that representations are collected from the exact homonym token span instead of relying on buggy mean-token approximations.

Implemented in:

- [models/models.py](/Users/juliahagen/LLM%20RSA%20Project/models/models.py:1)

Key functionality:

- exact target-span detection
- target-span mean pooling
- target-span last-token pooling
- sentence-mean baseline pooling
- paired-text activation path for WiC-style secondary validation
- environment-driven HF loading support for HPC deployment

Relevant environment variables already supported:

- `HUGGINGFACE_HUB_TOKEN`
- `HF_DEVICE_MAP`
- `HF_TRUST_REMOTE_CODE`

## 2. Distributed Semantic Profile Analysis

The within-model homonym analysis was implemented as a full layer-wise profile analysis.

Implemented in:

- [experiments/distributed_profiles.py](/Users/juliahagen/LLM%20RSA%20Project/experiments/distributed_profiles.py:1)
- [utils/profile_metrics.py](/Users/juliahagen/LLM%20RSA%20Project/utils/profile_metrics.py:1)

Current outputs include:

- GDV-by-layer per homonym
- probing F1-by-layer per homonym
- sentence-pooled baseline probe
- random-label baseline
- peak layer per homonym
- top-k layer windows
- profile sharpness
- semantic band estimation
- model-level peak distribution summaries
- across-model profile comparison
- cross-model CKA matrices

## 3. Probing Classifier

The probing classifier was rewritten to better support the conceptual goals of the project.

Implemented in:

- [probing/probing_classifier.py](/Users/juliahagen/LLM%20RSA%20Project/probing/probing_classifier.py:1)

Current features:

- repeated cross-validation
- grouped splits when provenance is available
- shuffled-label baseline
- fold-level result export

## 4. Homonym Data Generation

The old synthetic generation script was not strong enough for a clean rebuild. A new generation layer was added.

Implemented in:

- [data/homonym_generation.py](/Users/juliahagen/LLM%20RSA%20Project/data/homonym_generation.py:1)
- [data/homonym_seed_inventory.json](/Users/juliahagen/LLM%20RSA%20Project/data/homonym_seed_inventory.json:1)

Current generation pipeline:

- uses HF-hosted causal instruction models
- prompts in a JSON-return format
- validates exact target-word presence
- checks duplication
- applies heuristic sense anchors
- stores richer metadata per sense and family

Important note:

The current validation is much stronger than before, but it is still heuristic. For the full HPC study, manual spot-checking or a second-stage validator would still be advisable.

## 5. Data Preparation

The flattened synthetic dataset format was extended so that later analyses can use provenance-aware grouping.

Implemented in:

- [data/synthetic_data_preparation.py](/Users/juliahagen/LLM%20RSA%20Project/data/synthetic_data_preparation.py:66)

Added metadata includes:

- `family_id`
- `sample_id`
- `is_seed_sentence`
- `sample_index_within_family`
- `sense_id`
- `sense_name`
- `sense_gloss`
- `seed_sentence`
- `generation_model_id`

## 6. Central Model Registry

A central HF-only model catalog was added so the project no longer depends on ad hoc model lists spread across scripts.

Implemented in:

- [configs/model_registry.py](/Users/juliahagen/LLM%20RSA%20Project/configs/model_registry.py:1)
- [configs/pipeline_presets.py](/Users/juliahagen/LLM%20RSA%20Project/configs/pipeline_presets.py:1)

Default analysis models:

- `answerdotai/ModernBERT-large`
- `microsoft/deberta-v3-large`
- `FacebookAI/roberta-large`
- `FacebookAI/xlm-roberta-large`
- `Qwen/Qwen2.5-3B`
- `Qwen/Qwen2.5-7B`
- `mistralai/Mistral-Nemo-Base-2407`
- `allenai/OLMo-2-1124-7B`

Default generation models registered:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-Nemo-Instruct-2407`

These were chosen so the codebase relies on models that are available on Hugging Face and are more scientifically interesting than the previous legacy selection.

## 7. Pipeline Orchestration

The repo now has a single orchestrated pipeline layer instead of disconnected experimental scripts.

Implemented in:

- [pipeline/homonym_pipeline.py](/Users/juliahagen/LLM%20RSA%20Project/pipeline/homonym_pipeline.py:1)

Entry points:

- [run_full_pipeline.py](/Users/juliahagen/LLM%20RSA%20Project/run_full_pipeline.py:1)
- [run_generation.py](/Users/juliahagen/LLM%20RSA%20Project/run_generation.py:1)
- [run_h1.py](/Users/juliahagen/LLM%20RSA%20Project/run_h1.py:1)
- [run_experiments.py](/Users/juliahagen/LLM%20RSA%20Project/run_experiments.py:1)

Current intended use:

- `run_generation.py`: generate or load the homonym dataset
- `run_full_pipeline.py`: full homonym-first pipeline
- `run_h1.py`: convenience wrapper to the same full pipeline
- `run_experiments.py`: secondary WiC validation only

## Output Structure

Generated outputs are intended to live under:

- `results/`
- `results/distributed_profiles/`
- `results/cross_model_profiles/`
- `results/manifests/`
- `data/generated/`

Legacy outputs that are expected to be regenerated were moved to:

- [legacy/pre_hpc_refresh_2026-04-16](</Users/juliahagen/LLM RSA Project/legacy/pre_hpc_refresh_2026-04-16>)

## Current Limitations

The logic is in place, but the project has not yet been executed end-to-end at HPC scale from this environment.

Known current limitations:

- local environment here is missing runtime dependencies such as `pandas`
- HF models were not downloaded and run here
- the repo is not yet `git-clean` because the archival move appears as tracked deletions plus the new `legacy/` archive
- homonym dataset generation currently uses heuristic validation and should still be quality-checked before the full run

## HPC Handover Plan

### Step 1. Clean Repository State

Before moving to the cluster:

- create a cleanup commit for the new pipeline files
- commit the archival move into `legacy/`
- decide whether old tracked output files should remain versioned at all

### Step 2. Recreate Environment on HPC

Install the required environment with at least:

- `torch`
- `transformers`
- `huggingface_hub`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `statsmodels`

If you want to keep exact reproducibility, turn the HPC environment into a new explicit env file once it is working.

### Step 3. Configure HF Access

Set:

- `HUGGINGFACE_HUB_TOKEN` if a model requires authentication or gated access
- `HF_DEVICE_MAP` if you want model sharding
- `HF_TRUST_REMOTE_CODE=1` if a model needs custom code

### Step 4. First HPC Pilot Run

Do not start with the full study.

Start with:

- a reduced homonym set
- 2 or 3 analysis models
- reduced contexts per sense

Pilot goals:

- verify dataset generation
- verify exact homonym span extraction
- verify saved activation artifacts
- verify grouped probe splits
- verify result export and cross-model comparison

### Step 5. Expand Dataset

The project is currently prepared structurally, but the final scientific scale still needs to be built.

Recommended target:

- 30 to 40 homonyms
- 60 to 100 contexts per sense
- 8 to 12 models

The current seed inventory is only a starting point for this larger dataset.

### Step 6. Full Main Run

Once the pilot is stable:

- generate the larger homonym dataset
- run the full homonym profile suite
- run cross-model comparison
- run WiC validation only afterwards as a secondary sanity check

## Suggested Execution Order on HPC

1. `python run_generation.py`
2. `python run_full_pipeline.py`
3. `python run_experiments.py`

If you prefer a single main entry point, use:

1. `python run_h1.py`

## Practical HPC Adjustments Still Expected

The code logic should not need a conceptual rewrite anymore. The remaining HPC-side work should mainly be:

- adapting import paths if the cluster environment requires it
- adapting folder roots and scratch/output directories
- setting model cache locations
- setting job scripts and batch resource requests
- deciding whether to shard models or run one model per job

## Recommended Immediate Next Action

Before starting the full cluster migration:

- make one clean repository commit containing the new pipeline and the archived legacy outputs
- then perform one small HPC smoke test with `run_generation.py` and a reduced model subset

## One-Line Handover Summary

The repository now contains the full homonym-first distributed semantic profile pipeline, including HF-based homonym generation, exact target-span activation extraction, within-model analysis, across-model comparison, and secondary WiC validation; the remaining work is primarily HPC environment setup, scaling, and execution.
