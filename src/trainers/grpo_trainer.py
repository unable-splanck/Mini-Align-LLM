from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GRPOTrainerConfig:
    config: Dict[str, Any]


class GRPOTrainerScaffold:
    """Scaffold for future GRPO implementation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = GRPOTrainerConfig(config=config)

    def summary(self) -> str:
        data = self.config.config.get("data", {})
        return (
            "GRPO scaffold ready: "
            f"group_size={data.get('group_size')}, "
            f"prompt_file={data.get('prompt_file')}"
        )

