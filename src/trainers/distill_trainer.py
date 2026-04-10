from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DistillTrainerConfig:
    config: Dict[str, Any]


class DistillTrainerScaffold:
    """Scaffold for future distillation implementation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = DistillTrainerConfig(config=config)

    def summary(self) -> str:
        training = self.config.config.get("training", {})
        return (
            "Distill scaffold ready: "
            f"output_dir={training.get('output_dir')}, "
            f"epochs={training.get('num_train_epochs')}"
        )

