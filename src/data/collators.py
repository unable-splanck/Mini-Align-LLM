from typing import Dict, List

import torch


class SupervisedDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_length = max(feature["input_ids"].shape[0] for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            seq_len = feature["input_ids"].shape[0]
            pad_len = max_length - seq_len

            input_ids.append(
                torch.cat(
                    [feature["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]
                )
            )
            attention_mask.append(
                torch.cat([feature["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            labels.append(
                torch.cat([feature["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

