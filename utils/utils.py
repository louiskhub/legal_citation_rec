from collections import defaultdict
import os
import pickle
from typing import Tuple

from sklearn.model_selection import KFold, train_test_split
from transformers import DebertaForSequenceClassification, DebertaTokenizerFast

from config import DEV_SPLIT, TEST_SPLIT, SEED, VOCAB_FP


def evaluation_metrics(result):
    preds = result.predictions
    labels = result.label_ids
    recall_1 = 0
    recall_5 = 0
    recall_3 = 0
    recall_20 = 0
    count_dict = defaultdict(int)
    with open("./preds.txt", "w") as f:
        for i in range(0, len(preds)):
            raw_pred = list(preds[i])
            label = labels[i]
            count_dict[str(label)] += 1
            idx = sorted(
                range(len(raw_pred)), key=lambda sub: raw_pred[sub], reverse=True
            )[:20]
            f.write(str(label) + " " + ",".join([str(i) for i in idx]) + "\n")
            if label not in idx:
                continue
            elif label not in idx[:5]:
                recall_20 += 1
            elif label not in idx[:3]:
                recall_5 += 1
            elif label != idx[0]:
                recall_3 += 1
            else:
                recall_1 += 1

    recall_1_ratio = recall_1 / len(preds)
    recall_3 += recall_1
    recall_3_ratio = recall_3 / len(preds)
    recall_5 += recall_3
    recall_5_ratio = recall_5 / len(preds)
    recall_20 += recall_5
    recall_20_ratio = recall_20 / len(preds)

    return {
        "recall@1": recall_1_ratio,
        "recall@3": recall_3_ratio,
        "recall@5": recall_5_ratio,
        "recall@20": recall_20_ratio,
    }


def split_data(opinions_dir: str, cross_validation=True) -> None:
    """
    Split the data into train, dev, and test set.
    If cross_validation is True, split the data into 6 folds.
    The file names of the splits will be stored as txt files in the utils folder.
    """
    file_names = [
        f.split(".")[0] for f in os.listdir(opinions_dir) if f.lower().endswith(".pt")
    ]

    train_names, test_names = train_test_split(
        file_names,
        test_size=TEST_SPLIT,
        random_state=SEED,
        shuffle=True,
    )

    if cross_validation:
        kf = KFold(n_splits=6, shuffle=True, random_state=SEED)
        # tbd
    else:
        train_names, dev_names = train_test_split(
            train_names,
            test_size=DEV_SPLIT,
            random_state=SEED,
            shuffle=True,
        )
        with open("utils/data_split/train.txt", "w") as f:
            for name in train_names:
                f.write(name + "\n")
        with open("utils/data_split/dev.txt", "w") as f:
            for name in dev_names:
                f.write(name + "\n")
        with open("utils/data_split/test.txt", "w") as f:
            for name in test_names:
                f.write(name + "\n")


def load_vocab():
    with open(VOCAB_FP, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def init_model(embedding_len: int, n_classes: int) -> DebertaForSequenceClassification:
    model = DebertaForSequenceClassification.from_pretrained(
        "microsoft/deberta-base", num_labels=n_classes
    )
    model.resize_token_embeddings(embedding_len)
    return model


def init_tokenizer() -> Tuple[DebertaTokenizerFast, int, int]:
    tokenizer: DebertaTokenizerFast = DebertaTokenizerFast.from_pretrained(
        "microsoft/deberta-base"
    )
    tokenizer.add_tokens(["@pb@", "@cit@"])  # paragraphs + citations
    pb_id, cit_id = tokenizer.convert_tokens_to_ids(["@pb@", "@cit@"])
    return tokenizer, pb_id, cit_id
