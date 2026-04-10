import json
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.records = self._load_records()

    def _load_records(self) -> List[Dict[str, str]]:
        if not self.path.exists():
            raise FileNotFoundError(f"Preference dataset not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.records[index]

