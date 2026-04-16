# LLM-EEG Master Project for Neuroscience Lab at FAU 
---
## Neural and Artificial Correlates in Language Processing: A Comparative Study of LLMs and EEG data

---
Research Question:
"Do large language models (LLMs) display activation patterns that reflect the regional specialisation observed in the brain's language processing areas?"

## Current Pipeline

The codebase now supports a homonym-first rebuild with distributed semantic profiles.

- `run_generation.py` builds or reloads a semantically controlled homonym dataset from [data/homonym_seed_inventory.json](/Users/juliahagen/LLM%20RSA%20Project/data/homonym_seed_inventory.json).
- `run_full_pipeline.py` runs the full homonym pipeline: dataset generation, target-span activation extraction, within-model distributed profile analysis, and across-model profile comparison.
- `run_h1.py` is now a convenience wrapper around the same full pipeline.
- `run_experiments.py` remains the secondary WiC validation runner.

## Default HF Model Set

The default analysis catalog is defined in [configs/model_registry.py](/Users/juliahagen/LLM%20RSA%20Project/configs/model_registry.py) and currently uses HF-hosted models only:

- `answerdotai/ModernBERT-large`
- `microsoft/deberta-v3-large`
- `FacebookAI/roberta-large`
- `FacebookAI/xlm-roberta-large`
- `Qwen/Qwen2.5-3B`
- `Qwen/Qwen2.5-7B`
- `mistralai/Mistral-Nemo-Base-2407`
- `allenai/OLMo-2-1124-7B`

The default synthetic-data generator is `Qwen/Qwen2.5-7B-Instruct`, with `mistralai/Mistral-Nemo-Instruct-2407` registered as an additional HF-available generation option.
