from transformers import PreTrainedTokenizer
from typing import List, Tuple

class EnhancedWicPreprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, split: str = "train"):
        """
        Initialize the preprocessor.

        Args:
            tokenizer: A Hugging Face tokenizer instance.
            split (str): Dataset split to load ('train', 'validation', or 'test').
        """
        self.tokenizer = tokenizer
        self.split = split

    def load_data(self) -> Tuple[List[str], List[str], List[str], List[int]]:
        """
        Load the WiC dataset and extract its elements.

        Returns:
            texts1 (List[str]): The first sentence from each sample.
            texts2 (List[str]): The second sentence from each sample.
            targets (List[str]): The target word from each sample.
            labels (List[int]): The binary label.
        """
        data = load_wic_dataset(self.split)
        texts1, texts2, targets, labels = zip(*data)
        return list(texts1), list(texts2), list(targets), list(labels)

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize the given text using the provided tokenizer.
        
        Args:
            text (str): The text to tokenize.
        
        Returns:
            tokens (List[str]): A list of token strings.
        """
        return self.tokenizer.tokenize(text)

    def find_subword_indices(self, tokens: List[str], target_tokens: List[str]) -> List[int]:
        """
        Find the contiguous sequence of target_tokens in the tokenized text.
        
        Args:
            tokens (List[str]): Tokenized version of the sentence.
            target_tokens (List[str]): Tokenized version of the target word.
        
        Returns:
            indices (List[int]): List of indices corresponding to the target word's subwords.
                                  Returns an empty list if no exact match is found.
        """
        n = len(tokens)
        m = len(target_tokens)
        for i in range(n - m + 1):
            if tokens[i:i + m] == target_tokens:
                return list(range(i, i + m))
        # Optionally log a warning if no match is found.
        return []

    def process(self) -> Tuple[List[str], List[str], List[str], List[int], List[List[int]], List[List[int]]]:
        """
        Process the dataset: load data, tokenize sentences, and extract target token indices.
        
        Returns:
            texts1 (List[str]): Raw first sentences (can be useful later).
            texts2 (List[str]): Raw second sentences.
            targets (List[str]): The target word for each sample.
            labels (List[int]): The binary labels.
            indices1 (List[List[int]]): A list containing, for each sample, the indices of the target tokens in sentence1.
            indices2 (List[List[int]]): Similarly, for sentence2.
        """
        texts1, texts2, targets, labels = self.load_data()
        indices1 = []
        indices2 = []
        for t1, t2, target in zip(texts1, texts2, targets):
            # Tokenize the sentences.
            tokens1 = self.tokenize_text(t1)
            tokens2 = self.tokenize_text(t2)
            
            # Tokenize the target.
            target_tokens = self.tokenize_text(target)
            
            # Find the indices where the target tokens appear.
            idx1 = self.find_subword_indices(tokens1, target_tokens)
            idx2 = self.find_subword_indices(tokens2, target_tokens)
            
            indices1.append(idx1)
            indices2.append(idx2)
        return texts1, texts2, targets, labels, indices1, indices2
