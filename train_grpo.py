import argparse

from src.trainers.grpo_trainer import GRPOTrainerScaffold
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO scaffold.")
    parser.add_argument("--config", default="configs/grpo.yaml")
    args = parser.parse_args()

    logger = get_logger("train_grpo")
    config = load_yaml_config(args.config)
    set_seed(config.get("seed", 42))

    trainer = GRPOTrainerScaffold(config)
    logger.info(trainer.summary())
    logger.info("Next step: implement grouped sampling, relative rewards, and GRPO optimization.")


if __name__ == "__main__":
    main()

