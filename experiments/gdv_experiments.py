
import os
import re
import torch
from torch.cuda.amp import autocast
import pandas as pd
from dotenv import load_dotenv
from models import load_model_and_tokenizer, get_target_activations
from data import flatten_dataframe


def run_gdv_experiment(df, model_name):
    df_flat = flatten_dataframe(df)
    print("Flattened dataset:")
    print(df_flat.head())

    model, tokenizer = load_model_and_tokenizer(model_name)

    sentences = df_flat["sentence"].tolist()
    target_words = df_flat["word"].tolist()

    activations = get_target_activations(model, tokenizer, sentences, target_words, batch_size=4)