import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Dict


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_record(record: Dict[str, str]) -> Dict[str, str]:
    return {
        "instruction": str(record.get("instruction", "")).strip(),
        "input": str(record.get("input", "")).strip(),
        "output": str(record.get("output", "")).strip(),
    }


def split_records(records: List[Dict[str, str]], val_ratio: float, seed: int) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    return train_records, val_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT train/validation JSONL files.")
    parser.add_argument("--input-file", default="data/raw/demo_instructions.jsonl")
    parser.add_argument("--train-file", default="data/processed/sft_train.jsonl")
    parser.add_argument("--val-file", default="data/processed/sft_val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [normalize_record(record) for record in load_jsonl(Path(args.input_file))]
    records = [record for record in records if record["instruction"] and record["output"]]
    train_records, val_records = split_records(records, args.val_ratio, args.seed)

    save_jsonl(Path(args.train_file), train_records)
    save_jsonl(Path(args.val_file), val_records)

    print(f"Prepared {len(train_records)} train records -> {args.train_file}")
    print(f"Prepared {len(val_records)} validation records -> {args.val_file}")


if __name__ == "__main__":
    main()

