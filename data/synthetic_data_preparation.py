import pandas as pd 
from transformers import PreTrainedTokenizerFast
import re

def tokenize_text(text: str, tokenizer) -> list:
    """Tokenize a string using the given Hugging Face tokenizer."""
    return tokenizer.tokenize(text)

import re
from typing import List
from transformers import PreTrainedTokenizerBase

def find_target_token_indices(
    sentence: str,
    word: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[int]:
    """
    Find the token positions in `sentence` that exactly cover the first
    occurrence of `word`. We use offset_mapping (so we turn off all
    special tokens, and rely on character spans).
    """
    # 1) locate the first occurrence of the word (case-insensitive)
    low_sent = sentence.lower()
    low_word = word.lower()
    m = re.search(re.escape(low_word), low_sent)
    if not m:
        return []
    start_char, end_char = m.span()

    # 2) tokenize *only* with offsets
    enc = tokenizer(
        sentence,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"]  # list of (char_start, char_end) per token

    # 3) pick tokens whose entire span lies within that word
    idxs = [
        i
        for i, (s, e) in enumerate(offsets)
        if s >= start_char and e <= end_char
    ]
    return idxs




def process_sentence(sentence: str, word: str, tokenizer):
    """
    Tokenize `sentence` into subwords, tokenize `word` into subwords,
    then find the contiguous sublist match and return (tokens, indices).
    """
    tokens = tokenizer.tokenize(sentence)          # e.g. ["ĠThere", "’", "s", ...]
    target_tokens = tokenizer.tokenize(word)       # e.g. ["Ġbank"]
    # find target_tokens as a contiguous slice of tokens:
    n, m = len(tokens), len(target_tokens)
    for i in range(n - m + 1):
        if tokens[i : i + m] == target_tokens:
            return tokens, list(range(i, i + m))
    # no match → return empty indices
    return tokens, []


def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with columns "examples" (a list), "word", and "semantic_group_id",
    flatten the "examples" list so that each generated sentence becomes its own row.
    """
    rows = []
    for _, row in df.iterrows():
        word = row["word"]
        group_id = row["semantic_group_id"]
        for sent in row["examples"]:
            rows.append({"sentence": sent, "word": word, "semantic_group_id": group_id})
    return pd.DataFrame(rows)
