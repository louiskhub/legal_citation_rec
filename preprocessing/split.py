import os
from pathlib import Path
import logging

from sklearn.model_selection import KFold, train_test_split

from config import (
    DEV_SPLIT,
    SEED,
    TEST_SPLIT,
    VOCAB_SIZES,
    ICLOUD_FP,
    TEXT_FP,
    SPLIT_FP,
    LOGS_FP,
)


logging.basicConfig(
    filename=os.path.join(LOGS_FP, "split.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)


def split_data(vsize: int, suffix=".pt", cross_validation=False) -> None:
    """
    Split the data into train, dev, and test set.
    If cross_validation is True, split the data into 6 folds.
    The file names of the splits will be stored as txt files in the utils folder.
    """

    ddir: str = os.path.join(ICLOUD_FP, TEXT_FP, f"vocab_size_{vsize}")

    file_names = [
        f.removesuffix(suffix) for f in os.listdir(ddir) if f.lower().endswith(suffix)
    ]

    train_names, test_names = train_test_split(
        file_names,
        test_size=TEST_SPLIT,
        random_state=SEED,
        shuffle=True,
    )

    if cross_validation:  # to be done ...
        kf = KFold(n_splits=6, shuffle=True, random_state=SEED)
        print("cross validation is not implemented yet!")
    else:
        train_names, dev_names = train_test_split(
            train_names,
            test_size=DEV_SPLIT / (1 - TEST_SPLIT),  # recalc the ratio
            random_state=SEED,
            shuffle=True,
        )

        for set_type, set_names in zip(
            ["train", "dev", "test"],
            [train_names, dev_names, test_names],
        ):
            save_names(vsize, set_type, set_names)


def save_names(vsize: int, set_type: str, set_names: list[str]):
    fp = os.path.join(SPLIT_FP, f"vocab_size_{vsize}")
    Path(fp).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(fp, set_type + ".txt"), "w") as f:
        for name in set_names:
            f.write(name + "\n")

    logging.info(f"vocab_size_{vsize} {set_type} set is saved.")


if __name__ == "__main__":
    for vsize in VOCAB_SIZES:
        split_data(vsize=vsize)
