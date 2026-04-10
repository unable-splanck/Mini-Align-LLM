from dataclasses import dataclass
from typing import Any, Dict, Optional

from transformers import Trainer, TrainingArguments

from src.data.collators import SupervisedDataCollator
from src.data.sft_dataset import SFTDataset
from src.models.policy_model import load_policy_model
from src.models.tokenizer import load_tokenizer


@dataclass
class SFTTrainerBundle:
    trainer: Trainer
    train_dataset: SFTDataset
    eval_dataset: Optional[SFTDataset]


def build_sft_trainer(config: Dict[str, Any]) -> SFTTrainerBundle:
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]

    tokenizer = load_tokenizer(
        model_config["name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    model = load_policy_model(
        model_config["name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", False),
        use_fp16=model_config.get("use_fp16", False),
    )

    train_dataset = SFTDataset(
        path=data_config["train_file"],
        tokenizer=tokenizer,
        prompt_template=data_config["prompt_template"],
        max_length=data_config["max_length"],
    )
    eval_path = data_config.get("validation_file")
    eval_dataset = None
    if eval_path:
        eval_dataset = SFTDataset(
            path=eval_path,
            tokenizer=tokenizer,
            prompt_template=data_config["prompt_template"],
            max_length=data_config["max_length"],
        )

    collator = SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id)

    args = TrainingArguments(
        output_dir=training_config["output_dir"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        logging_steps=training_config["logging_steps"],
        save_strategy=training_config["save_strategy"],
        eval_strategy=training_config["evaluation_strategy"],
        max_grad_norm=training_config["max_grad_norm"],
        fp16=model_config.get("use_fp16", False),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    return SFTTrainerBundle(
        trainer=trainer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
