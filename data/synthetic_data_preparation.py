import pandas as pd


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
