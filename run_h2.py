import gc
import logging
import os
import sys
import traceback

import pandas as pd

from experiments import run_gdv_experiment
from utils.file_manager import FileManager
from utils.hpc import configure_hpc_runtime


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
configure_hpc_runtime()


def main():
    sys.path.append(os.path.abspath(".."))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("OFFLOAD_DIR:", os.environ.get("OFFLOAD_DIR"))

    file_manager = FileManager()
    del file_manager

    data_file = "data/synthetic_data_h2.pkl"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File {data_file} not found.")
    df = pd.read_pickle(data_file)
    logging.info(
        "Loaded %s: %d word-sense pairs across %d unique words",
        data_file, len(df), df['word'].nunique(),
    )

    model_configs = {
        "answerdotai/ModernBERT-large":         {"model_type": "default"},
        "microsoft/deberta-v3-large":           {"model_type": "default"},
        "FacebookAI/roberta-large":             {"model_type": "default"},
        "FacebookAI/xlm-roberta-large":         {"model_type": "default"},
        "Qwen/Qwen2.5-3B":                      {"model_type": "default"},
        "Qwen/Qwen2.5-7B":                      {"model_type": "default"},
        "mistralai/Mistral-Nemo-Base-2407":     {"model_type": "default"},
        "allenai/OLMo-2-1124-7B":              {"model_type": "default"},
    }

    for model_name, cfg in model_configs.items():
        logging.info("Running GDV experiment for %s", model_name)
        try:
            run_gdv_experiment(df, model_name, model_type=cfg["model_type"])
        except Exception as exc:
            logging.error("Failed to run %s: %s", model_name, exc)
            logging.debug(traceback.format_exc())
        finally:
            gc.collect()


if __name__ == "__main__":
    main()
