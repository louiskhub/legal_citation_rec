from __future__ import annotations
import os
import logging
import argparse

from transformers import Trainer, TrainingArguments
import torch

from dataset_low_ram import CitationDataset
from utils.utils import init_tokenizer, init_model, load_vocab, evaluation_metrics

logging.basicConfig(level=logging.INFO)


def custom_data_collator(features):
    batch = {}
    # 'input_ids' and 'labels' should match the argument names of your model's forward method
    batch["input_ids"] = torch.stack([item[0] for item in features])
    batch["labels"] = torch.stack([item[1] for item in features])
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer Task")
    parser.add_argument("-d", "--data_gcs_path", type=str)
    parser.add_argument("-g", "--gradient_checkpointing", type=bool, default=False)
    args = vars(parser.parse_args())

    v = load_vocab()

    tokenizer, _, cit_id = init_tokenizer()
    model = init_model(len(tokenizer), n_classes=len(v))

    train_ds = CitationDataset(
        data_dir=os.path.join(os.path.join(args["data_gcs_path"], "data")),
        set_type="train",
        vocab_size=len(v),
    )
    dev_ds = CitationDataset(
        data_dir=os.path.join(os.path.join(args["data_gcs_path"], "data")),
        set_type="dev",
        vocab_size=len(v),
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.join(args["data_gcs_path"], "outputs")),
        evaluation_strategy="steps",
        eval_steps=500,
        do_train=True,
        do_eval=True,
        fp16=False,
        save_strategy="steps",
        save_steps=50,
        per_device_train_batch_size=192,
        per_device_eval_batch_size=192,
        logging_first_step=False,
        logging_steps=9,
        learning_rate=1e-4,
        save_total_limit=5,
        dataloader_num_workers=1,
        gradient_accumulation_steps=3,
        gradient_checkpointing=args["gradient_checkpointing"],
        num_train_epochs=50,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=evaluation_metrics,
        data_collator=custom_data_collator,
    )

    trainer.train()
    trainer.save_model()
