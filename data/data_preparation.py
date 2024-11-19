from dataclasses import dataclass
from typing import Callable, List, Tuple, Any
from transformers import PreTrainedTokenizer

@dataclass
class DatasetPreprocessor:
    """
    A class for preparing datasets for probing tasks.

    Attributes:
        dataset_loader (Callable): Function to load the dataset.
        tokenizer (PreTrainedTokenizer): Tokenizer for text preprocessing.
        preprocess_fn (Callable): Function to preprocess dataset samples.
    """
    dataset_loader: Callable
    tokenizer: PreTrainedTokenizer
    preprocess_fn: Callable

    def load_and_prepare(self, split: str = "train") -> Tuple[List[str], List[int]]:
        """
        Load and prepare the dataset for probing.

        Args:
            split (str): The dataset split to load ('train', 'validation', 'test').

        Returns:
            Tuple[List[str], List[int]]: Prepared input texts and labels.
        """
        # Load dataset using the provided loader function
        raw_data = self.dataset_loader(split)

        # Preprocess dataset samples
        texts, labels = self.preprocess_fn(raw_data, self.tokenizer)
        return texts, labels

