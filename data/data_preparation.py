from dataclasses import dataclass
from typing import Callable, List, Tuple, Any
from transformers import PreTrainedTokenizer

@dataclass
class DatasetPreprocessor:
    """
    A class to dynamically load and preprocess datasets for probing experiments.
    """
    def __init__(self, tokenizer, dataset_name):
        """
        Initialize the DatasetPreprocessor.

        Args:
            tokenizer: Tokenizer for preprocessing text.
            dataset_name (str): Name of the dataset (e.g., "wic").
        """
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name.lower()
        self.dataset_loader, self.preprocess_fn = self._get_loader_and_preprocessor()

    def _get_loader_and_preprocessor(self):
        """
        Dynamically select the loader and preprocessor based on dataset_name.
        """
        if self.dataset_name == "wic":
            from data.data_loaders import load_wic_dataset, preprocess_wic
            return load_wic_dataset, preprocess_wic
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def load_and_prepare(self, split="train"):
        """
        Load and preprocess the dataset.

        Args:
            split (str): Dataset split to load (e.g., "train").

        Returns:
            Tuple[List[str], List[int]]: Processed texts and labels.
        """
        raw_data = self.dataset_loader(split)
        return self.preprocess_fn(raw_data, self.tokenizer)
