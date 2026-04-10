from pathlib import Path
from typing import Iterable, Mapping


def write_case_study(path: str, rows: Iterable[Mapping[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Case Study\n\n")
        for row in rows:
            handle.write(f"## Prompt\n{row.get('prompt', '')}\n\n")
            handle.write(f"### Base\n{row.get('base', '')}\n\n")
            handle.write(f"### SFT\n{row.get('sft', '')}\n\n")
            handle.write(f"### Notes\n{row.get('notes', '')}\n\n")

