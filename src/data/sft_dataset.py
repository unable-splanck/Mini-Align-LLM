import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, prompt_template: str, max_length: int) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.records = self._load_records()

    def _load_records(self) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        if not self.path.exists():
            raise FileNotFoundError(f"SFT dataset not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        prompt = self.prompt_template.format(
            instruction=record.get("instruction", "").strip(),
            input=record.get("input", "").strip(),
        )
        response = record.get("output", "").strip()

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        eos_token_id = self.tokenizer.eos_token_id

        input_ids = prompt_ids + response_ids + ([eos_token_id] if eos_token_id is not None else [])
        labels = [-100] * len(prompt_ids) + response_ids + ([eos_token_id] if eos_token_id is not None else [])

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

