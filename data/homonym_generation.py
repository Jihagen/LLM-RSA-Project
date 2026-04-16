import json
import os
import pickle
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_pandas():
    import pandas as pd

    return pd


@dataclass
class SenseDefinition:
    sense_id: str
    sense_name: str
    sense_gloss: str
    seed_sentence: str
    positive_anchors: List[str] = field(default_factory=list)
    negative_anchors: List[str] = field(default_factory=list)


@dataclass
class HomonymDefinition:
    word: str
    senses: List[SenseDefinition]


class HuggingFaceCausalGenerator:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        load_args = {}

        if torch.cuda.is_available():
            load_args["low_cpu_mem_usage"] = True

        device_map = os.environ.get("HF_DEVICE_MAP")
        if device_map:
            load_args["device_map"] = device_map

        if os.environ.get("HF_TRUST_REMOTE_CODE", "0") == "1":
            load_args["trust_remote_code"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **load_args)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            **load_args,
        )
        if "device_map" not in load_args:
            self.model = self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate linguistically natural and semantically controlled English sentences "
                    "for homonym research. Always follow the requested JSON schema exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompt: str) -> str:
        formatted_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        if generated_text.startswith(formatted_prompt):
            return generated_text[len(formatted_prompt) :].strip()
        return generated_text.strip()


def load_homonym_inventory(path: str) -> List[HomonymDefinition]:
    with open(path, "r") as file:
        payload = json.load(file)

    inventory: List[HomonymDefinition] = []
    for item in payload:
        senses = [
            SenseDefinition(
                sense_id=sense["sense_id"],
                sense_name=sense["sense_name"],
                sense_gloss=sense["sense_gloss"],
                seed_sentence=sense["seed_sentence"],
                positive_anchors=list(sense.get("positive_anchors", [])),
                negative_anchors=list(sense.get("negative_anchors", [])),
            )
            for sense in item["senses"]
        ]
        inventory.append(HomonymDefinition(word=item["word"], senses=senses))
    return inventory


def _normalize_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def _contains_exact_word(sentence: str, word: str) -> bool:
    pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", flags=re.IGNORECASE)
    return bool(pattern.search(sentence))


def _word_count(sentence: str) -> int:
    return len(re.findall(r"\b\w+\b", sentence))


def _extract_json_payload(raw_text: str) -> Optional[object]:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", candidate)
        candidate = re.sub(r"\n```$", "", candidate).strip()

    for open_char, close_char in (("[", "]"), ("{", "}")):
        start = candidate.find(open_char)
        end = candidate.rfind(close_char)
        if start >= 0 and end > start:
            snippet = candidate[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _parse_generated_sentences(raw_text: str) -> List[str]:
    payload = _extract_json_payload(raw_text)
    sentences: List[str] = []

    if isinstance(payload, dict):
        for key in ("sentences", "examples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                sentences.append(item)
            elif isinstance(item, dict) and "sentence" in item:
                sentences.append(str(item["sentence"]))
        return [_normalize_sentence(sentence) for sentence in sentences if sentence]

    fallback_lines = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\s*[-*\d\.\)]\s*", "", line).strip()
        if cleaned:
            fallback_lines.append(cleaned)
    return [_normalize_sentence(sentence) for sentence in fallback_lines if sentence]


def _sentence_passes_anchors(sentence: str, sense: SenseDefinition) -> bool:
    lowered = sentence.lower()
    if sense.positive_anchors:
        if not any(anchor.lower() in lowered for anchor in sense.positive_anchors):
            return False
    if sense.negative_anchors:
        if any(anchor.lower() in lowered for anchor in sense.negative_anchors):
            return False
    return True


def validate_generated_sentence(
    sentence: str,
    word: str,
    sense: SenseDefinition,
    seen_sentences: Sequence[str],
    min_words: int = 6,
    max_words: int = 40,
) -> Tuple[bool, Optional[str]]:
    normalized = _normalize_sentence(sentence)

    if not normalized:
        return False, "empty"
    if normalized.lower() in {item.lower() for item in seen_sentences}:
        return False, "duplicate"
    if not _contains_exact_word(normalized, word):
        return False, "missing_exact_target_word"
    if _word_count(normalized) < min_words:
        return False, "too_short"
    if _word_count(normalized) > max_words:
        return False, "too_long"
    if not _sentence_passes_anchors(normalized, sense):
        return False, "anchor_validation_failed"
    return True, None


def build_generation_prompt(
    word: str,
    sense: SenseDefinition,
    requested_count: int,
    existing_sentences: Sequence[str],
) -> str:
    existing_preview = list(existing_sentences[:10])
    return (
        "Generate semantically controlled English sentences for a homonym study.\n"
        f"Target word: {word}\n"
        f"Sense ID: {sense.sense_id}\n"
        f"Sense name: {sense.sense_name}\n"
        f"Sense gloss: {sense.sense_gloss}\n"
        f"Seed sentence: {sense.seed_sentence}\n"
        f"Positive anchors: {sense.positive_anchors}\n"
        f"Negative anchors: {sense.negative_anchors}\n"
        f"Already accepted examples to avoid repeating: {existing_preview}\n\n"
        "Requirements:\n"
        f"- Return exactly {requested_count} new sentences.\n"
        f"- Every sentence must contain the exact surface form '{word}'.\n"
        "- Keep the same sense as the seed sentence.\n"
        "- Use natural, varied syntax and lexical contexts.\n"
        "- Do not number the sentences.\n"
        "- Do not explain your reasoning.\n"
        '- Return valid JSON in the form {"sentences": ["...", "..."]}.\n'
    )


def _sense_to_record(
    word: str,
    sense: SenseDefinition,
    semantic_group_id: int,
    examples: Sequence[str],
    generation_model_id: str,
    target_examples_per_sense: int,
) -> Dict[str, object]:
    return {
        "word": word,
        "sense_id": sense.sense_id,
        "sense_name": sense.sense_name,
        "sense_gloss": sense.sense_gloss,
        "semantic_group_id": semantic_group_id,
        "family_id": f"{word}_{sense.sense_id}",
        "seed_sentence": sense.seed_sentence,
        "examples": list(examples),
        "positive_anchors": list(sense.positive_anchors),
        "negative_anchors": list(sense.negative_anchors),
        "generation_model_id": generation_model_id,
        "target_examples_per_sense": int(target_examples_per_sense),
        "validated_example_count": int(len(examples)),
    }


def generate_homonym_dataset(
    inventory: Sequence[HomonymDefinition],
    generator: HuggingFaceCausalGenerator,
    examples_per_sense_target: int,
    minimum_examples_per_sense: int,
    request_size_per_generation_call: int,
    max_generation_rounds: int,
    random_seed: int = 42,
):
    pd = _load_pandas()
    rng = random.Random(random_seed)

    records: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    semantic_group_id = 0

    for homonym in inventory:
        for sense in homonym.senses:
            accepted_examples = [sense.seed_sentence]
            rejected_counts: Dict[str, int] = {}

            for _round in range(max_generation_rounds):
                remaining = examples_per_sense_target - len(accepted_examples)
                if remaining <= 0:
                    break

                request_count = min(request_size_per_generation_call, remaining)
                prompt = build_generation_prompt(
                    word=homonym.word,
                    sense=sense,
                    requested_count=request_count,
                    existing_sentences=accepted_examples,
                )
                raw_text = generator.generate(prompt)
                candidates = _parse_generated_sentences(raw_text)
                rng.shuffle(candidates)

                for candidate in candidates:
                    is_valid, reason = validate_generated_sentence(
                        sentence=candidate,
                        word=homonym.word,
                        sense=sense,
                        seen_sentences=accepted_examples,
                    )
                    if is_valid:
                        accepted_examples.append(candidate)
                        if len(accepted_examples) >= examples_per_sense_target:
                            break
                    elif reason is not None:
                        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

            record = _sense_to_record(
                word=homonym.word,
                sense=sense,
                semantic_group_id=semantic_group_id,
                examples=accepted_examples,
                generation_model_id=generator.model_id,
                target_examples_per_sense=examples_per_sense_target,
            )
            records.append(record)
            summary_rows.append(
                {
                    "word": homonym.word,
                    "sense_id": sense.sense_id,
                    "validated_example_count": len(accepted_examples),
                    "target_examples_per_sense": examples_per_sense_target,
                    "minimum_examples_per_sense": minimum_examples_per_sense,
                    "meets_minimum_threshold": len(accepted_examples) >= minimum_examples_per_sense,
                    "rejected_counts": rejected_counts,
                }
            )
            semantic_group_id += 1

    df = pd.DataFrame(records)
    summary = {
        "generation_model_id": generator.model_id,
        "n_homonyms": len(inventory),
        "n_senses": len(records),
        "examples_per_sense_target": examples_per_sense_target,
        "minimum_examples_per_sense": minimum_examples_per_sense,
        "sense_rows": summary_rows,
    }
    return df, summary


def save_homonym_dataset(
    df,
    summary: Dict[str, object],
    pickle_path: str,
    jsonl_path: str,
    summary_path: str,
) -> None:
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    df.to_pickle(pickle_path)
    with open(jsonl_path, "w") as file:
        for row in df.to_dict(orient="records"):
            file.write(json.dumps(row) + "\n")
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)


def load_or_generate_homonym_dataset(
    seed_inventory_path: str,
    generation_model_id: str,
    output_pickle_path: str,
    output_jsonl_path: str,
    output_summary_path: str,
    examples_per_sense_target: int,
    minimum_examples_per_sense: int,
    request_size_per_generation_call: int,
    max_generation_rounds: int,
    random_seed: int = 42,
    force_regenerate: bool = False,
):
    pd = _load_pandas()

    if os.path.exists(output_pickle_path) and not force_regenerate:
        return pd.read_pickle(output_pickle_path)

    inventory = load_homonym_inventory(seed_inventory_path)
    generator = HuggingFaceCausalGenerator(
        model_id=generation_model_id,
    )
    df, summary = generate_homonym_dataset(
        inventory=inventory,
        generator=generator,
        examples_per_sense_target=examples_per_sense_target,
        minimum_examples_per_sense=minimum_examples_per_sense,
        request_size_per_generation_call=request_size_per_generation_call,
        max_generation_rounds=max_generation_rounds,
        random_seed=random_seed,
    )
    save_homonym_dataset(
        df=df,
        summary=summary,
        pickle_path=output_pickle_path,
        jsonl_path=output_jsonl_path,
        summary_path=output_summary_path,
    )
    return df


__all__ = [
    "SenseDefinition",
    "HomonymDefinition",
    "HuggingFaceCausalGenerator",
    "build_generation_prompt",
    "generate_homonym_dataset",
    "load_homonym_inventory",
    "load_or_generate_homonym_dataset",
    "save_homonym_dataset",
    "validate_generated_sentence",
]
