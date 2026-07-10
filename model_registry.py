"""
Canonical model name/alias lists, shared by run_study.py and the hypotheses/
runners so the "which models does each hypothesis run on" list lives in one
place instead of several copies that can silently drift out of sync.

No heavy imports here (no torch/transformers) — safe to import at module
load time without forcing model-loading dependencies onto lightweight
entry points (e.g. --help).
"""

ALL_MODELS = [
    "answerdotai/ModernBERT-large",
    "microsoft/deberta-v3-large",
    "FacebookAI/roberta-large",
    "FacebookAI/xlm-roberta-large",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "mistralai/Mistral-Nemo-Base-2407",
    "allenai/OLMo-2-1124-7B",
]

MODEL_ALIASES = {
    "deberta":    "microsoft/deberta-v3-large",
    "roberta":    "FacebookAI/roberta-large",
    "xlm":        "FacebookAI/xlm-roberta-large",
    "modernbert": "answerdotai/ModernBERT-large",
    "qwen3b":     "Qwen/Qwen2.5-3B",
    "qwen7b":     "Qwen/Qwen2.5-7B",
    "mistral":    "mistralai/Mistral-Nemo-Base-2407",
    "olmo":       "allenai/OLMo-2-1124-7B",
}
