import os
import json
from collections import OrderedDict
from typing import Tuple

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EvalPrediction,
)

from config import (
    VOCAB_FP,
    MODEL_NAME,
    LR,
)


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


def init_model(
    embedding_len: int, n_classes: int
) -> DistilBertForSequenceClassification:
    """Initializes the DistilBERT model.
    The token embeddings are resized to updated (@cit@, @pb@) vocabulary size.)
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes
    )
    model.resize_token_embeddings(embedding_len)  # type: ignore
    return model  # type: ignore


def init_tokenizer() -> Tuple[DistilBertTokenizerFast, int, int]:
    """Initializes the DistilBERT tokenizer.
    Resize the vocabulary to include the new tokens (@cit@, @pb@).
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
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
