import argparse

from src.trainers.sft_trainer import build_sft_trainer
from src.utils.checkpoint import save_metadata
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised fine-tuning model.")
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()

    logger = get_logger("train_sft")
    config = load_yaml_config(args.config)
    set_seed(config.get("seed", 42))

    bundle = build_sft_trainer(config)
    logger.info("Loaded %s training samples.", len(bundle.train_dataset))
    if bundle.eval_dataset is not None:
        logger.info("Loaded %s validation samples.", len(bundle.eval_dataset))

    train_result = bundle.trainer.train()
    bundle.trainer.save_model()
    bundle.trainer.save_state()

    save_metadata(
        config["training"]["output_dir"],
        {
            "train_samples": len(bundle.train_dataset),
            "eval_samples": 0 if bundle.eval_dataset is None else len(bundle.eval_dataset),
            "metrics": train_result.metrics,
        },
    )
    logger.info("SFT training complete. Model saved to %s", config["training"]["output_dir"])


if __name__ == "__main__":
    main()

