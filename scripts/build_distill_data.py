import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal distillation dataset scaffold.")
    parser.add_argument("--output-file", default="data/processed/distill_train.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demo_records = [
        {
            "instruction": "解释什么是奖励模型。",
            "input": "",
            "teacher_output": "奖励模型用于评估模型输出质量，为后续对齐优化提供打分信号。",
        }
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        for record in demo_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Distillation scaffold written to {output_path}")


if __name__ == "__main__":
    main()

