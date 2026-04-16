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
    Flatten the synthetic homonym dataframe into one row per sentence.

    In addition to the original sentence/word/semantic_group_id fields, the flattened
    representation carries provenance information that can later be used for grouped
    evaluation or leakage checks.
    """
    rows = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        word = row["word"]
        group_id = int(row["semantic_group_id"])
        examples = list(row["examples"])
        family_id = row["family_id"] if "family_id" in row else f"{word}_sense_{group_id}"

        for example_index, sent in enumerate(examples):
            rows.append(
                {
                    "sentence": sent,
                    "word": word,
                    "semantic_group_id": group_id,
                    "family_id": family_id,
                    "sense_id": row["sense_id"] if "sense_id" in row else None,
                    "sense_name": row["sense_name"] if "sense_name" in row else None,
                    "sense_gloss": row["sense_gloss"] if "sense_gloss" in row else None,
                    "seed_sentence": row["seed_sentence"] if "seed_sentence" in row else None,
                    "generation_model_id": row["generation_model_id"] if "generation_model_id" in row else None,
                    "sample_index_within_family": example_index,
                    "is_seed_sentence": example_index == 0,
                    "sample_id": f"{family_id}_{example_index}",
                    "row_position": row_index,
                }
            )
    return pd.DataFrame(rows)
