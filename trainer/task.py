import os
import logging
import argparse

from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Subset
import wandb

from utils import (
    init_tokenizer,
    evaluation_metrics,
    init_optimizer,
    get_datasets_from_wandb,
    init_model_from_wandb,
)
from keys import WANDB_API_KEY


logging.basicConfig(level=logging.INFO)

os.environ["WANDB_API_KEY"] = WANDB_API_KEY
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def custom_data_collator(features):
    batch = {}
    # 'input_ids' and 'labels' should match the argument names of your model
    batch["input_ids"] = torch.stack([item[0] for item in features])
    batch["labels"] = torch.stack([item[1] for item in features])
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer Task")
    parser.add_argument("-g", "--data_gcs_path", type=str)
    parser.add_argument("-c", "--checkpoint", type=int)
    parser.add_argument("-s", "--vocab_size", type=int, default=4287)
    parser.add_argument("-d", "--dataset_type", type=str, default="ordered")
    args = vars(parser.parse_args())
    vsize: int = args["vocab_size"]
    dataset_type: str = args["dataset_type"]

    datasets = get_datasets_from_wandb(dataset_type, vsize)
    wandb.login()

    tokenizer, _, cit_id = init_tokenizer()

    config = {
        "embedding_len": len(tokenizer),
        "n_classes": vsize,
        "model_name": "distilbert",
        "n_checkpoint": args["checkpoint"],
        "dataset_type": dataset_type,
        "base_fp": "",
    }

    model = init_model_from_wandb(**config)
    optimizer = init_optimizer(model)

    training_args = TrainingArguments(
        run_name=f"training-vsize{vsize}-{dataset_type}-singlefile",
        report_to="wandb",  # type: ignore
        output_dir=args["data_gcs_path"],
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        fp16=False,
        save_strategy="epoch",
        per_device_train_batch_size=64,  # 128
        per_device_eval_batch_size=64,  # 128
        logging_first_step=False,
        logging_steps=9,
        learning_rate=1e-4,
        save_total_limit=8,
        dataloader_num_workers=1,
        gradient_accumulation_steps=9,  # 3
        eval_accumulation_steps=3000,
        num_train_epochs=5,
    )

    smaller_dev_set = Subset(datasets["dev"], range(len(datasets["dev"]) // 4))
    del datasets["dev"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=smaller_dev_set,
        compute_metrics=evaluation_metrics,
        optimizers=(
            optimizer,
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1),
        ),
        data_collator=custom_data_collator,
    )

    trainer.train()
    trainer.save_model()
