# collator.py
from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class SFTCollator:
    tokenizer
    max_length: int = 2048

    def __call__(self, features: List[Dict]):
        texts = []
        for f in features:
            messages = [
                {"role": "system", "content": "You are a clinical NLP assistant."},
                {"role": "user", "content": f"{f['instruction']}\n\nINPUT:\n{f.get('input','')}"},
                {"role": "assistant", "content": str(f.get('output', ''))}
            ]
            txt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(txt)

        batch = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt",
            max_length=self.max_length
        )
        labels = batch["input_ids"].clone()
        batch["labels"] = labels
        return batch
