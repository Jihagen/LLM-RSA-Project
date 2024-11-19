from datasets import load_dataset
from typing import Callable, List, Tuple, Any
from transformers import PreTrainedTokenizer



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

def load_wikitext_dataset(split="train"):
    """
    Load the WikiText dataset and return non-empty text samples.
    """
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    return [{"text": line.strip()} for line in dataset["text"] if line.strip()]


def load_trec_dataset(split="train"):
    """
    Load the TREC dataset with coarse-grained labels.
    """
    dataset = load_dataset("trec", split=split)
    # Dynamically handle label types (e.g., label-fine or label-coarse)
    label_key = "coarse_label" if "coarse_label" in dataset.column_names else "fine_label"
    return [{"text": entry["text"], "label": entry[label_key]} for entry in dataset]


def load_commoncrawl_dataset(split="train"):
    path = f"data/commoncrawl/{split}.txt"  # Update with actual path
    with open(path, "r") as f:
        lines = f.readlines()
    return [{"text": line.strip()} for line in lines]

def load_story_dataset(split="train", config="2018"):
    """
    Load the Story Cloze Dataset with the specified configuration.
    Args:
        split (str): Dataset split to load (e.g., "train", "validation").
        config (str): Config name to specify the version of the dataset ("2016" or "2018").
    Returns:
        List[Dict]: List of dictionaries with context, endings, and labels.
    """
    dataset = load_dataset("story_cloze", config, split=split)

    # Transform data into context, ending, and label format
    processed_data = []
    for entry in dataset:
        processed_data.append({
            "context": entry["context"],          # Story context (first 4 sentences)
            "ending1": entry["ending0"],          # First candidate ending
            "ending2": entry["ending1"],          # Second candidate ending
            "label": entry["label"]               # 1 if ending1 is correct, 2 if ending2 is correct
        })

    return processed_data



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

def preprocess_wikitext(data, tokenizer):
    """
    Preprocess the WikiText dataset for probing experiments.
    Assign dummy labels since WikiText is unstructured text.
    """
    texts = [entry["text"] for entry in data]
    labels = [0] * len(texts)  # Dummy labels for unstructured text
    return texts, labels


def preprocess_trec(data, tokenizer):
    texts = [entry["text"] for entry in data]
    labels = [entry["label"] for entry in data]
    return texts, labels

def preprocess_commoncrawl(data, tokenizer):
    texts = [entry["text"] for entry in data]
    labels = [0] * len(data)  # Dummy labels
    return texts, labels

def preprocess_story(data, tokenizer):
    """
    Preprocess the Story Cloze Dataset for probing experiments.
    Args:
        data (List[Dict]): List of context, endings, and labels.
        tokenizer: Tokenizer to preprocess the text.
    Returns:
        Tuple[List[str], List[int]]: Preprocessed text pairs and labels.
    """
    texts = []
    labels = []

    for entry in data:
        # Format as "context + candidate ending"
        context = entry["context"]
        ending1 = entry["ending1"]
        ending2 = entry["ending2"]

        # Generate input pairs for both endings
        texts.append(f"{context} {ending1}")
        texts.append(f"{context} {ending2}")

        # Label the correct ending (convert 1/2 to 0/1)
        labels.append(1 if entry["label"] == 1 else 0)
        labels.append(0 if entry["label"] == 1 else 1)

    return texts, labels
