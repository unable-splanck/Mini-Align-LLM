import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal preference-style dataset scaffold.")
    parser.add_argument("--output-file", default="data/processed/pref_train.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demo_records = [
        {
            "prompt": "解释什么是监督微调。",
            "chosen": "监督微调是用高质量指令数据继续训练基础模型，使它更符合具体任务和人类偏好。",
            "rejected": "监督微调就是随便继续训练一下模型。",
        }
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        for record in demo_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Preference scaffold written to {output_path}")


if __name__ == "__main__":
    main()

