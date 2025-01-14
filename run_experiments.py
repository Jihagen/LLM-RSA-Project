# run_experiments.py

import logging
import sys
import os
from utils.file_manager import FileManager
from models import load_model_and_tokenizer
from experiments import run_layer_identification_experiment
import traceback
import torch 
import gc 

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Add project root to sys.path
    sys.path.append(os.path.abspath(".."))
    
    file_manager = FileManager()

    # List of LLMs to test
    llms_to_test = [
       # "bert-base-uncased",
        # "distilbert-base-uncased",
        #"roberta-base",
        #"xlm-roberta-base",
        # "gpt2",
        "gpt-neo-1.3B",
        "gpt-j-6B",
        "decapoda-research/llama-7b-hf",
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B",
        "tiiuae/falcon-7b",
        "bigscience/bloom-560m",
        "t5-base",
    ]

    # Dataset and split details
    dataset_name = "wic"
    split = "train"

    # Iterate through the list of LLMs and run the experiment
    for model_name in llms_to_test:
        logging.info(f"Testing model: {model_name}")
        try:
            model, tokenizer = load_model_and_tokenizer(model_name)
            logging.debug(f"Loaded model and tokenizer for {model_name}")
            run_layer_identification_experiment(model, tokenizer, dataset_name=dataset_name, split=split)
        except Exception as e:
            logging.error(f"Failed to test model {model_name}: {e}")
            logging.debug(traceback.format_exc())
        
        finally:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("Cleared GPU cache")
            
            # Delete model and tokenizer references
            del model, tokenizer
            
            # Trigger garbage collection
            gc.collect()
            logging.debug("Triggered garbage collection")

if __name__ == "__main__":
    main()