import logging
import sys
import os
import torch
import gc
import traceback
from utils.file_manager import FileManager
from models import load_model_and_tokenizer
from experiments import run_layer_identification_experiment

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Add project root to sys.path
    sys.path.append(os.path.abspath(".."))
    
    file_manager = FileManager()

    # Define model configurations with a model_type parameter.
    # Here we have:
    # - "default": No special handling.
    # - "auth": Requires authentication (token).
    # - "encoder-decoder": Reserved for later special handling.
    model_configs = {
        #"bert-base-uncased": {"model_type": "default"},
        #"distilbert-base-uncased": {"model_type": "default"},
        #"roberta-base": {"model_type": "default"},
        #"xlm-roberta-base": {"model_type": "default"},
        #"gpt2": {"model_type": "default"},
        "EleutherAI/gpt-neo-1.3B": {"model_type": "default"},  
        "EleutherAI/gpt-j-6B": {"model_type": "default"},       
        "decapoda-research/llama-7b-hf": {"model_type": "auth"},
        "meta-llama/Llama-2-7b-hf": {"model_type": "auth"},
        "mistralai/Mistral-7B": {"model_type": "auth"},
        "tiiuae/falcon-7b": {"model_type": "default"},
        "bigscience/bloom-560m": {"model_type": "default"},
        #"t5-base": {"model_type": "encoder-decoder"}, 
    }

    llms_to_test = list(model_configs.keys())

    # Dataset and split details
    dataset_name = "wic"
    split = "train"

    for model_name in llms_to_test:
        model, tokenizer = None, None  # Initialize to avoid UnboundLocalError
        logging.info(f"Testing model: {model_name}")
        try:
            model_type = model_configs[model_name]["model_type"]
            model, tokenizer = load_model_and_tokenizer(model_name, model_type=model_type)
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

            
if __name__ == "__main__":
    main()
