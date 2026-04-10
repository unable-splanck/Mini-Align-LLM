import json
from pathlib import Path
from typing import Any, Dict


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_metadata(path: str, metadata: Dict[str, Any]) -> None:
    output_dir = ensure_output_dir(path)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

