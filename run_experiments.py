import gc
import logging
import os
import sys
import traceback

from data.synthetic import SyntheticDataset
from experiments import run_layer_identification_experiment
from models.models import TokenProbeModel, load_tokenizer
from utils.file_manager import FileManager
from utils.hpc import configure_hpc_runtime, model_device


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
configure_hpc_runtime()


def main():
    sys.path.append(os.path.abspath(".."))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("OFFLOAD_DIR:", os.environ.get("OFFLOAD_DIR"))

    file_manager = FileManager()
    del file_manager

    model_configs = {
        "answerdotai/ModernBERT-large": {"model_type": "default"},
        "microsoft/deberta-v3-large": {"model_type": "default"},
        "FacebookAI/roberta-large": {"model_type": "default"},
        "FacebookAI/xlm-roberta-large": {"model_type": "default"},
        "Qwen/Qwen2.5-3B": {"model_type": "default"},
        "Qwen/Qwen2.5-7B": {"model_type": "default"},
        "mistralai/Mistral-Nemo-Base-2407": {"model_type": "default"},
        "allenai/OLMo-2-1124-7B": {"model_type": "default"},
    }

    llms_to_test = list(model_configs.keys())
    dataset_name = "wic"
    split = "train"

    for model_name in llms_to_test:
        model = None
        tokenizer = None
        logging.info("Testing model: %s", model_name)
        try:
            model_type = model_configs[model_name]["model_type"]
            model = TokenProbeModel(model_name, model_type=model_type)
            tokenizer = load_tokenizer(model_name, model_type=model_type)
            dataset = SyntheticDataset('data/synthetic.pkl', tokenizer)
            del dataset
            device = model_device(model)
            logging.info("Model %s is running on %s", model_name, device)
            run_layer_identification_experiment(model, tokenizer, dataset_name=dataset_name, split=split)
        except Exception as exc:
            logging.error("Failed to test model %s: %s", model_name, exc)
            logging.debug(traceback.format_exc())
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()


if __name__ == "__main__":
    main()
