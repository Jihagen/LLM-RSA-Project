import os
import re
from typing import List, Tuple
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd


# Load environment variables from the .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Pass the API key to the OpenAI LLM instance explicitly
llm = OpenAI(api_key=openai_api_key, temperature=0.7)

# Define a prompt template that instructs the LLM
prompt_template = PromptTemplate(
    input_variables=["input_sentence", "word"],
    template=(
        "You are given the word '{word}', which is a homonym ({word} carries a different meaning depending on the context)"
        "The sentence: '{input_sentence}' defines the context. Based on this context generate 4 new sentences that use "
        "the exact word '{word}' in the same semantic context. Each of the sentences has to contain the word '{word}'."
        "Make sure to only return a list of strings separated by commas without any numbering, introductory or closing remarks."
    )
)

# Create the LLMChain with the prompt template and LLM
chain = LLMChain(llm=llm, prompt=prompt_template)

def find_common_word(sentence1: str, sentence2: str) -> str:
    """
    Finds a common word between two sentences.
    Tokenizes by word (ignoring punctuation) and compares in lowercase.
    Returns an empty string if no common word is found.
    """
    tokens1 = set(re.findall(r'\w+', sentence1.lower()))
    tokens2 = set(re.findall(r'\w+', sentence2.lower()))
    common = tokens1.intersection(tokens2)
    return list(common)[0] if common else ""

def generate_similar_sentences(input_sentence: str, word: str) -> List[str]:
    """
    Generates 4 new similar sentences using a given word in the same semantic context,
    using the LangChain LLMChain.
    Assumes the LLM returns sentences separated by newlines.
    """
    result = chain.run(input_sentence=input_sentence, word=word)
    # Split the returned text into sentences, stripping extra whitespace.
    sentences = [line.strip() for line in result.split("\n") if line.strip()]
    return sentences

def process_sample(sample: Tuple[str, str, int], semantic_group_start: int = 0) -> List[Tuple[List[str], str, int]]:
    """
    Process one WiC sample.
    
    If label == 0, splits the sample into two semantic subclasses (one per sentence)
    and generates 4 additional sentences for each.
    If no common word is found between the two sentences, the sample is skipped.
    
    Returns a list of tuples in the form: ([examples], common_word, semantic_group_id).
    """
    sentence1, sentence2, label = sample
    common_word = find_common_word(sentence1, sentence2)
    # If no common word is found, skip this sample.
    if not common_word:
        print("Skipping sample due to no common word found.")
        return []
    
    results = []
    group_id = semantic_group_start

    if label == 0:
        for sent in [sentence1, sentence2]:
            additional = generate_similar_sentences(sent, common_word)
            examples = [sent] + additional
            results.append((examples, common_word, group_id))
            group_id += 1
    else:
        # For non-zero labels, treat the sample as a single semantic subclass.
        additional = generate_similar_sentences(sentence1, common_word)
        examples = [sentence1] + additional
        results.append((examples, common_word, group_id))

    return results

if __name__ == "__main__":
    # Starting sample:
    sample = ("There's a bank in the river", "She went to the bank to open a new account", 0)
    data_file = "data/synthetic_data_h1.pkl"
    
    # Check if the file already exists.
    if os.path.exists(data_file):
        df_existing = pd.read_pickle(data_file)
        # Start new semantic group IDs from max existing id + 1.
        new_start = df_existing["semantic_group_id"].max() + 1
        print(f"Existing dataset found. Starting semantic group id from {new_start}.")
    else:
        df_existing = pd.DataFrame(columns=["examples", "word", "semantic_group_id"])
        new_start = 0
        print("No existing dataset found. Creating a new one.")
    
    # Process the sample with the appropriate starting semantic group id.
    enriched_samples = process_sample(sample, semantic_group_start=new_start)
    
    # If we generated new samples, append them and save the DataFrame.
    if enriched_samples:
        df_new = pd.DataFrame(enriched_samples, columns=["examples", "word", "semantic_group_id"])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Optionally set the index name to "semantic_group_id"
        df_combined.index.name = "semantic_group_id"
        # Save the DataFrame to file.
        df_combined.to_pickle(data_file)
        print(f"New enriched samples appended. Total samples: {len(df_combined)}")
    else:
        print("No new samples were generated.")