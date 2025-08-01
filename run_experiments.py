import logging
import sys
import os
import torch
import gc
import traceback
import shutil  # For deleting directories
from utils.file_manager import FileManager
from models import load_model_and_tokenizer
from experiments import run_layer_identification_experiment

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Add project root to sys.path
    sys.path.append(os.path.abspath(".."))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    
    file_manager = FileManager()

    # Define model configurations with a model_type parameter.
    # Here we have:
    # - "default": No special handling.
    # - "auth": Requires authentication (token).

    model_configs = {
         "bert-base-uncased": {"model_type": "default"},
         "distilbert-base-uncased": {"model_type": "default"},
         "roberta-base": {"model_type": "default"},
         "xlm-roberta-base": {"model_type": "default"},
         "gpt2": {"model_type": "default"},
         "EleutherAI/gpt-neo-1.3B": {"model_type": "default"},
         "EleutherAI/gpt-j-6B": {"model_type": "default"},
         "meta-llama/Llama-2-7b-hf": {"model_type": "auth"},
         "mistralai/Mistral-7B-v0.1": {"model_type": "auth"},
         "mistralai/Mistral-7B-v0.3": {"model_type": "auth"},
         "tiiuae/falcon-7b": {"model_type": "default"},
         "bigscience/bloom-560m": {"model_type": "default"},
    }

    llms_to_test = list(model_configs.keys())
    from data.synthetic import SyntheticDataset
    from transformers import AutoTokenizer

    # Dataset and split details
    dataset_name = "wic"
    split = "train"

    for model_name in llms_to_test:
        model, tokenizer = None, None  # Initialize to avoid UnboundLocalError
        logging.info(f"Testing model: {model_name}")
        try:
            from models.models import TokenProbeModel
            model = TokenProbeModel(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            dataset = SyntheticDataset('data/synthetic.pkl', tokenizer)
            device = next(model.parameters()).device
            logging.info(f"Model {model_name} is running on {device}")
            logging.debug(f"Loaded model and tokenizer for {model_name}")
            run_layer_identification_experiment(model, tokenizer, dataset_name=dataset_name, split=split)
        except Exception as e:
            logging.error(f"Failed to test model {model_name}: {e}")
            logging.debug(traceback.format_exc())
        finally:
            if model is not None:
                del model, tokenizer
                gc.collect()
            
            # Remove the cached model files to free up disk space.
            # Hugging Face models are cached under "~/.cache/huggingface/hub/models--{model_name with '/' replaced by '--'}"
            cache_base = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"))
            # Build the expected directory name (this is how the hub typically names model folders)
            model_cache_dir = os.path.join(cache_base, f"models--{model_name.replace('/', '--')}")
            if os.path.exists(model_cache_dir):
                try:
                    shutil.rmtree(model_cache_dir)
                    logging.info(f"Deleted cache for model {model_name} at {model_cache_dir}")
                except Exception as ex:
                    logging.error(f"Failed to delete cache for model {model_name} at {model_cache_dir}: {ex}")

if __name__ == "__main__":
    main()
