import os
from pathlib import Path
import logging

from sklearn.model_selection import KFold, train_test_split
import torch

from config import (
    DEV_SPLIT,
    SEED,
    TEST_SPLIT,
    VOCAB_SIZES,
    TEXT_FP,
    LOGS_FP,
)


logging.basicConfig(
    filename=os.path.join(LOGS_FP, "split.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)


def split_data(vsize: int, cross_validation=False, shuffle=False) -> None:
    """
    Split the data into train, dev, and test set.
    If cross_validation is True, split the data into 6 folds.
    The file names of the splits will be stored as txt files in the utils folder.
    """
    
    contexts: torch.Tensor = torch.load(
        os.path.join(TEXT_FP, "preprocessed", f"size_{vsize}_contexts.pt")
    )
    labels: torch.Tensor = torch.load(os.path.join(TEXT_FP, "preprocessed", f"size_{vsize}_labels.pt"))

    x_train, x_test, y_train, y_test = train_test_split(
        contexts,
        labels,
        test_size=TEST_SPLIT,
        random_state=SEED,
        shuffle=shuffle,
    )
    del contexts
    del labels

    if cross_validation:  # to be done ...
        kf = KFold(n_splits=6, shuffle=True, random_state=SEED)
        print("cross validation is not implemented yet!")
    else:
        x_train, x_dev, y_train, y_dev = train_test_split(
            x_train,
            y_train,
            test_size=DEV_SPLIT / (1 - TEST_SPLIT),  # recalc the ratio
            random_state=SEED,
            shuffle=shuffle,
        )

        for set_type, ds in zip(
            ["train", "dev", "test"],
            [(x_train, y_train), (x_dev, y_dev), (x_test, y_test)],
        ):
            fp = os.path.join(TEXT_FP, "ordered", f"vocab_size_{vsize}", set_type)
            Path(fp).mkdir(parents=True, exist_ok=True)
            torch.save(ds[0], os.path.join(fp, "inputs.pt"))
            torch.save(ds[1], os.path.join(fp, "labels.pt"))
            logging.info(f"vocab_size_{vsize} {set_type} set is saved.")


if __name__ == "__main__":
    #for vsize in VOCAB_SIZES:
    split_data(vsize=4287)
