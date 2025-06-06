import pickle
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    """
    Loads a pre-generated synthetic dataset from a pickle file.  
    Each example should include:
      - sentence: str
      - label: int or str
      - homonym_idx: int (character index or word index of homonym)
      - tokenized: (optional) pre-tokenized outputs
    """
    def __init__(self, pkl_path: str, tokenizer, max_length: int = 128):
        """
        Args:
            pkl_path: Path to the synthetic.pkl file.
            tokenizer: A HuggingFace tokenizer for model input.
            max_length: Maximum sequence length for padding/truncation.
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.examples = []
        for entry in data:
            sentence = entry['sentence']
            label = entry['label']
            # the original homonym word index in the sentence
            word_idx = entry['homonym_word_index']

            # tokenize but keep track of token-to-word mapping
            tok = tokenizer(
                sentence,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt',
                return_offsets_mapping=True,
            )

            # Find token positions corresponding to the homonym word
            offsets = tok.offset_mapping[0].tolist()
            homonym_token_positions = []
            # assuming entry also has character start/end of homonym, else reconstruct
            start_char, end_char = entry['homonym_char_span']
            for idx, (st, en) in enumerate(offsets):
                if st >= start_char and en <= end_char:
                    homonym_token_positions.append(idx)

            self.examples.append({
                'input_ids': tok.input_ids[0],
                'attention_mask': tok.attention_mask[0],
                'label': label,
                'homonym_positions': homonym_token_positions,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'input_ids': ex['input_ids'],
            'attention_mask': ex['attention_mask'],
            'label': ex['label'],
            'homonym_positions': ex['homonym_positions'],
        }
