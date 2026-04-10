import argparse

from src.trainers.ppo_trainer import PPOTrainerScaffold
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO scaffold.")
    parser.add_argument("--config", default="configs/ppo.yaml")
    args = parser.parse_args()

    logger = get_logger("train_ppo")
    config = load_yaml_config(args.config)
    set_seed(config.get("seed", 42))

    trainer = PPOTrainerScaffold(config)
    logger.info(trainer.summary())
    logger.info("Next step: implement rollout, reward computation, advantage estimation, and PPO updates.")


if __name__ == "__main__":
    main()

