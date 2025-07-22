import os
import re
import json
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=openai_api_key, temperature=0.7)

# Prompt template for generating new sentences; count is dynamic
prompt_template = PromptTemplate(
    input_variables=["input_sentence", "word", "count"],
    template=(
        "You are given the word '{word}', which is a homonym. "
        "The sentence: '{input_sentence}' provides the context. "
        "Generate {count} new sentences using the exact word '{word}' in this same sense. "
        "Return a comma-separated list only."
    )
)
chain = LLMChain(llm=llm, prompt=prompt_template)


def find_common_word(sentence1: str, sentence2: str) -> Optional[str]:
    tokens1 = set(re.findall(r'\w+', sentence1.lower()))
    tokens2 = set(re.findall(r'\w+', sentence2.lower()))
    common = tokens1.intersection(tokens2)
    return next(iter(common), None)


def generate_similar_sentences(input_sentence: str, word: str, count: int) -> List[str]:
    """
    Use LLMChain to generate `count` new sentences for `word` in context of `input_sentence`.
    """
    response = chain.run(input_sentence=input_sentence, word=word, count=count)
    # Split by commas and clean
    candidates = [s.strip() for s in response.split(",") if s.strip()]
    return candidates[:count]


def process_pair(sentence: str, word: str, group_id: int, count: int) -> Tuple[List[str], str, int]:
    """
    Process one sense: seed sentence + generate `count` new variants.
    """
    variants = generate_similar_sentences(sentence, word, count)
    return [sentence] + variants, word, group_id


def main(
    seed_samples: List[Tuple[str, str, str, str, int]],
    examples_per_sense: int = 10,
    data_file: str = "data/synthetic_data_controlled.pkl"
):
    """
    seed_samples: list of tuples (word, sent1, sent2, label) where label=0 for two-sense homonym.
    Generates and saves structured dataset.
    """
    # Load or init DataFrame
    if os.path.exists(data_file):
        df = pd.read_pickle(data_file)
        start_gid = df["semantic_group_id"].max() + 1
        print(f"Existing data found. Starting group ID at {start_gid}.")
    else:
        df = pd.DataFrame(columns=["examples", "word", "semantic_group_id"])
        start_gid = 0
        print("Creating new dataset.")

    all_entries = []
    gid = start_gid
    for word, s1, s2, label in seed_samples:
        # Determine common word automatically or use provided
        common = find_common_word(s1, s2) or word
        if label == 0:
            # Two-sense homonym: produce two groups
            for sentence in (s1, s2):
                entry = process_pair(sentence, common, gid, examples_per_sense)
                all_entries.append(entry)
                gid += 1
        else:
            # Single sense grouping (label !=0): combine both seeds then variants
            combined_seed = s1  # use first sentence
            entry = process_pair(combined_seed, common, gid, examples_per_sense * 2)
            all_entries.append(entry)
            gid += 1

    if all_entries:
        df_new = pd.DataFrame(all_entries, columns=["examples", "word", "semantic_group_id"])
        df = pd.concat([df, df_new], ignore_index=True)
        df.index.name = "semantic_group_id"
        df.to_pickle(data_file)
        print(f"Appended {len(df_new)} semantic groups. Total samples: {len(df)}.")
    else:
        print("No new samples generated.")

if __name__ == "__main__":
    # Define your 5 homonym words with two example sentences each
    seed_samples = [
        ("bank", "There's a bank in the river.", "She went to the bank to open a new account.", 0),
        ("bat", "The bat flew out of the cave.", "He used a bat to hit the baseball.", 0),
        ("bark", "The dog began to bark loudly.", "The tree's bark was rough to the touch.", 0),
        ("pupil", "The pupil in the eye dilated.", "The school pupil studied diligently.", 0),
        ("spring", "The metal coil is a spring.", "They looked forward to spring after winter.", 0),
    ]
    main(seed_samples=seed_samples, examples_per_sense=10)
