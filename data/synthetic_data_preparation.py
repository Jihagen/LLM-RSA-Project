import pandas as pd 

def tokenize_text(text: str, tokenizer) -> list:
    """Tokenize a string using the given Hugging Face tokenizer."""
    return tokenizer.tokenize(text)

def find_subword_indices(tokens: list, target_tokens: list) -> list:
    """
    Find the contiguous sequence of target_tokens in the token list.
    Returns a list of indices if found, or an empty list if not found.
    """
    n = len(tokens)
    m = len(target_tokens)
    for i in range(n - m + 1):
        if tokens[i:i + m] == target_tokens:
            return list(range(i, i + m))
    return []

def process_sentence(sentence: str, target: str, tokenizer) -> (list, list):
    """
    Tokenize the sentence and the target word; then find which token indices
    in the sentence correspond to the target word. Returns (tokens, target_indices)
    """
    tokens = tokenize_text(sentence, tokenizer)
    target_tokens = tokenize_text(target, tokenizer)
    indices = find_subword_indices(tokens, target_tokens)
    return tokens, indices

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
