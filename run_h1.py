import logging
import sys
import os
import torch
import gc
import traceback
import shutil  # For deleting directories
from utils.file_manager import FileManager
from models import load_model_and_tokenizer
from experiments import run_gdv_experiment


import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Add project root to sys.path
    sys.path.append(os.path.abspath(".."))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    
    file_manager = FileManager()
    
    
    data_file = "data/synthetic_data_h1.pkl"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File {data_file} not found.")
    df = pd.read_pickle(data_file)
    
    model_configs = {
         "bert-base-uncased": {"model_type": "default"},
         "distilbert-base-uncased": {"model_type": "default"},
         "gpt2": {"model_type": "default"},
         "EleutherAI/gpt-neo-1.3B": {"model_type": "default"},
         "EleutherAI/gpt-j-6B": {"model_type": "default"},
         "meta-llama/Llama-2-7b-hf": {"model_type": "auth"},
         "mistralai/Mistral-7B-v0.3": {"model_type": "auth"},
         "tiiuae/falcon-7b": {"model_type": "default"},
         "bigscience/bloom-560m": {"model_type": "default"},}

    llms_to_test = list(model_configs.keys())

    # Dataset and split details
    dataset_name = "wic"
    split = "train"

    for model_name in llms_to_test:
       # model_type = model_configs[model_name]["model_type"]
        logging.debug(f"Loaded model and tokenizer for {model_name}")
        run_gdv_experiment(df, model_name) 



if __name__ == "__main__":
   main()