import os
import logging

import torch

from config import TEXT_FP, LOGS_FP, DOWNSIZED_VOCAB_SIZES

logging.basicConfig(
    filename=os.path.join(LOGS_FP, "unite_preprocessed_texts.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)


def unite_texts(text_dir: str, vsize: int) -> None:
    """Unite the preprocessed chunks of size 20k into a single tensor."""

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

    # make sure that inputs and labels match each other
    assert len(label_fnames) == len(input_fnames)

    # allocate memory for the final, large tensors
    labels: torch.Tensor = torch.empty((0,), dtype=torch.int16)
    contexts: torch.Tensor = torch.empty((0, 256), dtype=torch.int16)

    for input_f, label_f in zip(input_fnames, label_fnames):
        label_batch: torch.Tensor = torch.load(os.path.join(text_dir, label_f))
        input_batch: torch.Tensor = torch.load(os.path.join(text_dir, input_f))

        # make sure that inputs and labels match each other
        assert label_batch.shape[0] == input_batch.shape[0]

        labels = torch.cat([labels, label_batch])
        contexts = torch.vstack([contexts, input_batch])

    # save the final tensors in the same directory but with a different naming scheme
    # enables easy deletion of the old tensor chunks from CLI
    torch.save(labels, os.path.join(text_dir, f"size_{vsize}_labels.pt"))
    torch.save(contexts, os.path.join(text_dir, f"size_{vsize}_contexts.pt"))

    logging.info(f"Finished uniting texts for vocab size {vsize}.")


if __name__ == "__main__":
    # Earlier we stored the preprocessed text data in chunks of 20k samples.
    # Now we concatenate them incrementally to a single large tensor.

    for vsize in (4287,) + DOWNSIZED_VOCAB_SIZES:
        unite_texts(os.path.join(TEXT_FP, "preprocessed"), vsize)
