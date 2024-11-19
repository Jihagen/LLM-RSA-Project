from datasets import load_dataset
from typing import Callable, List, Tuple, Any
from transformers import PreTrainedTokenizer

"""WIC DATASET LOADER AND PREPROCESSOR"""

def load_wic_dataset(split: str = "train") -> List[Tuple[str, str, int]]:
    """
    Load the WiC dataset using the Hugging Face datasets library.

    Args:
        split (str): Dataset split to load ('train', 'validation', or 'test').

    Returns:
        List[Tuple[str, str, int]]: [(sentence1, sentence2, label), ...].
    """
    dataset = load_dataset("super_glue", "wic", split=split)
    return [
        (entry["sentence1"], entry["sentence2"], entry["label"])
        for entry in dataset
    ]

def preprocess_wic(data: List[Tuple[str, str, int]], tokenizer: PreTrainedTokenizer) -> Tuple[List[str], List[int]]:
    """
    Preprocess WiC dataset samples.

    Args:
        data (List[Tuple[str, str, int]]): Raw WiC data [(sentence1, sentence2, label), ...].
        tokenizer (PreTrainedTokenizer): Tokenizer for preprocessing.

    Returns:
        Tuple[List[str], List[int]]: Tokenized input texts and labels.
    """
    texts = [f"{s1} [SEP] {s2}" for s1, s2, _ in data]
    labels = [label for _, _, label in data]
    return texts, labels



"""DEFINE DATASET LOADER AND PREPROCESSOR"""