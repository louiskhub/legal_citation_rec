from __future__ import annotations
import os
import logging
import argparse

from transformers import Trainer, TrainingArguments
import torch

from dataset_single_file import CitationDataset
from utils import init_tokenizer, init_model, evaluation_metrics, init_optimizer

logging.basicConfig(level=logging.INFO)


def custom_data_collator(features):
    batch = {}
    # 'input_ids' and 'labels' should match the argument names of your model
    batch["input_ids"] = torch.stack([item[0] for item in features])
    batch["labels"] = torch.stack([item[1] for item in features])
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer Task")
    parser.add_argument("-d", "--data_gcs_path", type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=105)
    parser.add_argument("-g", "--gradient_checkpointing", type=bool, default=False)
    args = vars(parser.parse_args())
    vsize: int = args["vocab_size"]

    tokenizer, _, cit_id = init_tokenizer()
    model = init_model(len(tokenizer), n_classes=vsize)
    optimizer = init_optimizer(model)

    train_ds = CitationDataset(
        data_dir=os.path.join(args["data_gcs_path"], "text"),  # gs://legal_citation_rec
        set_type="train",
        vsize=vsize,
    )
    dev_ds = CitationDataset(
        data_dir=os.path.join(args["data_gcs_path"], "text"),  # gs://legal_citation_rec
        set_type="dev",
        vsize=vsize,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args["data_gcs_path"], "outputs"),
        evaluation_strategy="steps",
        eval_steps=2,
        do_train=True,
        do_eval=True,
        fp16=False,
        save_strategy="steps",
        save_steps=250,
        per_device_train_batch_size=4,  # 128
        per_device_eval_batch_size=4,  # 128
        logging_first_step=False,
        logging_steps=9,
        learning_rate=1e-4,
        save_total_limit=5,
        dataloader_num_workers=1,
        # gradient_accumulation_steps=9, # 3
        # gradient_checkpointing=True,
        num_train_epochs=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=evaluation_metrics,
        optimizers=(
            optimizer,
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1),
        ),
        data_collator=custom_data_collator,
    )

    trainer.train()
    trainer.save_model()
