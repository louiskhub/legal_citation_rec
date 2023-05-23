import os
import logging

import torch

from config import TEXT_FP, LOGS_FP, VOCAB_SIZES

logging.basicConfig(
    filename=os.path.join(LOGS_FP, "unite_preprocessed_texts.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)


def unite_texts(text_dir: str, vsize: int) -> None:
    input_fnames: list[str] = sorted(
        [
            fname
            for fname in os.listdir(text_dir)
            if fname.endswith(f"_inputs_size_{vsize}.pt")
        ]
    )

    label_fnames: list[str] = sorted(
        [
            fname
            for fname in os.listdir(text_dir)
            if fname.endswith(f"_labels_size_{vsize}.pt")
        ]
    )

    assert len(label_fnames) == len(input_fnames)

    labels: torch.Tensor = torch.empty((0,), dtype=torch.int16)
    contexts: torch.Tensor = torch.empty((0, 256), dtype=torch.int16)

    for input_f, label_f in zip(input_fnames, label_fnames):
        label_batch: torch.Tensor = torch.load(os.path.join(text_dir, label_f))
        input_batch: torch.Tensor = torch.load(os.path.join(text_dir, input_f))

        assert label_batch.shape[0] == input_batch.shape[0]

        labels = torch.cat([labels, label_batch])
        contexts = torch.vstack([contexts, input_batch])

        # os.remove(os.path.join(text_dir, label_f))
        # os.remove(os.path.join(text_dir, input_f))
        # logging.info(f"Deleted {input_f} and {label_f}.")

    torch.save(labels, os.path.join(TEXT_FP, "preprocessedv2", f"size_{vsize}_labels.pt"))
    torch.save(
        contexts, os.path.join(TEXT_FP, "preprocessedv2", f"size_{vsize}_contexts.pt")
    )

    logging.info(f"Finished uniting texts for vocab size {vsize}.")


if __name__ == "__main__":
    #for vsize in VOCAB_SIZES:
    dir_fp: str = os.path.join(TEXT_FP, "preprocessed")
    unite_texts(dir_fp, 4287)
