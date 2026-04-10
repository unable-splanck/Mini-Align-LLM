import argparse

from src.trainers.distill_trainer import DistillTrainerScaffold
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distillation scaffold.")
    parser.add_argument("--config", default="configs/distill.yaml")
    args = parser.parse_args()

    logger = get_logger("train_distill")
    config = load_yaml_config(args.config)
    set_seed(config.get("seed", 42))

    trainer = DistillTrainerScaffold(config)
    logger.info(trainer.summary())
    logger.info("Next step: implement teacher generation, student targets, and distillation loss.")


if __name__ == "__main__":
    main()

