import os
import json
from collections import OrderedDict
from typing import Tuple

import torch
import wandb
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EvalPrediction,
)

from config import (
    VOCAB_FP,
    BASE_DISTILBERT,
    LR,
    TRAIN_SPLIT,
    TEST_SPLIT,
    DEV_SPLIT,
    CONTEXT_SIZE,
    FORCASTING_SIZE,
)
from dataset_single_file import CitationDataset


def evaluation_metrics(result: EvalPrediction):
    recall_1 = 0
    recall_5 = 0
    recall_3 = 0
    recall_20 = 0

    for pred, label in zip(result.predictions, result.label_ids):
        top_20_preds = pred.argsort()[-20:][::-1]

        if label not in top_20_preds:
            continue
        elif label not in top_20_preds[:5]:
            recall_20 += 1
        elif label not in top_20_preds[:3]:
            recall_5 += 1
        elif label != top_20_preds[0]:
            recall_3 += 1
        else:
            recall_1 += 1

    recall_1_ratio = recall_1 / len(result.predictions)
    recall_3 += recall_1
    recall_3_ratio = recall_3 / len(result.predictions)
    recall_5 += recall_3
    recall_5_ratio = recall_5 / len(result.predictions)
    recall_20 += recall_5
    recall_20_ratio = recall_20 / len(result.predictions)

    return {
        "recall@1": recall_1_ratio,
        "recall@3": recall_3_ratio,
        "recall@5": recall_5_ratio,
        "recall@20": recall_20_ratio,
    }


def load_vocab(size: int) -> OrderedDict:
    """Load the (new) vocabulary of the given size."""
    with open(os.path.join(VOCAB_FP, f"size_{size}.json"), "r") as f:
        vocab = json.load(f, object_pairs_hook=OrderedDict)
    return vocab


def init_model_from_file(
    embedding_len: int,
    n_classes: int,
    model_name: str = "distilbert",
    n_checkpoint: int = 0,
    dataset_type: str = "",
    base_fp: str = "",
) -> DistilBertForSequenceClassification:
    """Initializes the DistilBERT model.
    The token embeddings are resized to updated (@cit@, @pb@) vocabulary size.)
    """
    if n_checkpoint > 0:
        model = DistilBertForSequenceClassification.from_pretrained(
            os.path.join(
                base_fp, model_name, f"vsize_{n_classes}", f"checkpoint-{n_checkpoint}"
            ),
            num_labels=n_classes,
        )
    else:
        model = DistilBertForSequenceClassification.from_pretrained(
            BASE_DISTILBERT, num_labels=n_classes
        )
    model.resize_token_embeddings(embedding_len)  # type: ignore
    return model  # type: ignore


def init_model_from_wandb(
    embedding_len: int,
    n_classes: int,
    model_name: str = "distilbert",
    n_checkpoint: int = 0,
    dataset_type: str = "",
    base_fp: str = "",
) -> DistilBertForSequenceClassification:
    """Initializes the DistilBERT model.
    The token embeddings are resized to updated (@cit@, @pb@) vocabulary size.)
    """

    with wandb.init(
        project="legal-citation-rec",
        entity="advanced-nlp",
        job_type="init-model",
    ) as run:  # type: ignore
        model_artifact = run.use_artifact(
            f"{model_name}-vsize{n_classes}-dataset_type_{dataset_type}-ckpt_{n_checkpoint}:latest"
        )
        model_fp = model_artifact.download()

    model = DistilBertForSequenceClassification.from_pretrained(
        model_fp,
        num_labels=n_classes,
    )
    model.resize_token_embeddings(embedding_len)  # type: ignore
    return model  # type: ignore


def upload_model_to_wandb(config: dict) -> None:
    """Uploads the model to weights and biases artifacts."""

    with wandb.init(
        project="legal-citation-rec",
        entity="advanced-nlp",
        job_type="upload-model",
        config=config,
    ) as run:  # type: ignore
        config = wandb.config

        model_artifact = wandb.Artifact(
            f"{config['model_name']}-vsize{config['n_classes']}-dataset_type_{config['dataset_type']}-ckpt_{config['n_checkpoint']}",
            type="model",
            description=f"Standart configuration of {config['model_name']}",
            metadata=dict(config),
        )

        if config["n_checkpoint"] > 0:
            fp: str = os.path.join(
                config["base_fp"],
                config["model_name"],
                f'vsize_{config["n_classes"]}',
                f'checkpoint-{config["n_checkpoint"]}',
            )
        else:
            fp: str = BASE_DISTILBERT

        model_artifact.add_dir(fp)

        run.log_artifact(model_artifact)


def init_tokenizer() -> Tuple[DistilBertTokenizerFast, int, int]:
    """Initializes the DistilBERT tokenizer.
    Resize the vocabulary to include the new tokens (@cit@, @pb@).
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_DISTILBERT)
    tokenizer.add_tokens(["@pb@", "@cit@"])  # paragraphs + citations
    pb_id, cit_id = tokenizer.convert_tokens_to_ids(["@pb@", "@cit@"])
    return tokenizer, pb_id, cit_id


def init_optimizer(model: DistilBertForSequenceClassification) -> torch.optim.AdamW:
    """Initializes the optimizer for the DistilBERT model.
    Weight decay is set to 0.0 to avoid decaying the bias and LayerNorm weights.
    Used to overwrite the default huggingface scheduler.
    """

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)


def upload_datasets_to_wandb(
    dataset_type: str, vsize: int, model_name: str = "DistilBERT"
) -> None:
    """Uploads the datasets to weights and biases artifacts."""

    fp_ds = f"data/text/{dataset_type}"

    datasets = {
        set_type: CitationDataset(
            data_dir=fp_ds,
            set_type=set_type,
            vsize=vsize,
        )
        for set_type in (
            "train",
            "dev",
            "test",
        )
    }

    with wandb.init(
        project="legal-citation-rec",
        entity="advanced-nlp",
        job_type="upload-dataset",
    ) as run:  # type: ignore
        processed_data = wandb.Artifact(
            f"bva-vsize{vsize}-{dataset_type}-singlefile",
            type="dataset",
            description=f"Preprocessed BCA corpus with vocab size {vsize} in a single file.",
            metadata={
                "vsize": vsize,
                "dataset_type": dataset_type,
                "n_samples": [len(ds) for ds in datasets.values()],
                "tokenizer": model_name,
                "train_split": TRAIN_SPLIT,
                "dev_split": DEV_SPLIT,
                "test_split": TEST_SPLIT,
                "context_size": CONTEXT_SIZE,
                "forcasting_size": FORCASTING_SIZE,
            },
        )

        for name, ds in datasets.items():
            with processed_data.new_file(name + ".pt", mode="wb") as f:
                torch.save(ds, f)

        run.log_artifact(processed_data)


def get_datasets_from_wandb(
    dataset_type: str,
    vsize: int,
    version: str = "latest",
) -> dict[str, CitationDataset]:
    """Downloads the datasets from weights and biases artifacts."""

    with wandb.init(
        project="legal-citation-rec",
        entity="advanced-nlp",
        job_type="get-dataset",
    ) as run:  # type: ignore
        stored_artifact = wandb.use_artifact(
            f"bva-vsize{vsize}-{dataset_type}-singlefile:{version}"
        )
        downloaded_artifact = stored_artifact.download()

        datasets = {
            set_type: torch.load(os.path.join(downloaded_artifact, set_type + ".pt"))
            for set_type in ("train", "dev", "test")
        }

    return datasets
