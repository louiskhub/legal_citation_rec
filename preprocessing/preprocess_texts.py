import os
import json
import logging
import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from transformers import DistilBertTokenizerFast

from config import (
    LOGS_FP,
    TEXT_FP,
    ICLOUD_FP,
    CONTEXT_SIZE,
    FORCASTING_SIZE,
    VOCAB_SIZES,
)
from utils import init_tokenizer, load_vocab


logging.basicConfig(
    filename=os.path.join(LOGS_FP, "preprocess_texts.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)


class OfflinePreprocessor:
    def __init__(
        self,
        tokenizer: DistilBertTokenizerFast,
        citation_token_id: int,
        vocab: OrderedDict,
        context_size: int = CONTEXT_SIZE,
        forcasting_size: int = FORCASTING_SIZE,
    ) -> None:
        self.file_names = self.get_filenames()
        self.tokenizer = tokenizer
        self.citation_token_id = citation_token_id
        self.vocab = vocab
        self.context_size = context_size
        self.forcasting_size = forcasting_size
        self.fp = os.path.join(ICLOUD_FP, "data", "text", f"vocab_size_{len(vocab)}")
        Path(self.fp).mkdir(parents=True, exist_ok=True)

    def get_filenames(self) -> list[str]:
        """
        Returns all filenames (without suffix) of the original dataset.
        """
        return [
            f.removesuffix(".json")
            for f in os.listdir(os.path.join(TEXT_FP, "original"))
            if f.lower().endswith(".json")
        ]

    def run(self, start_at: int) -> None:
        """
        Runs the preprocessing pipeline, starting at the given index.
        Files are saved as .pt files in the data/text folder. The file names
        are the same as the original JSON files.
        """

        for fidx, fname in enumerate(self.file_names[start_at:], start=start_at):
            encoded_text, vocab_indices = self.load_data_from_disk(fname)
            encoded_text = self.pad(encoded_text)
            context_windows, labels = self.select_data_samples(
                encoded_text, vocab_indices
            )

            # only save if we could extract data based on the given vocab
            if labels:
                data_sample = {
                    "inputs": torch.stack(context_windows),
                    "labels": torch.tensor(labels, dtype=torch.int16),
                }

                torch.save(
                    data_sample,
                    os.path.join(self.fp, fname + ".pt"),
                )

            self.log_progress(fidx, fname)

    def load_data_from_disk(self, file_name: str) -> tuple[list[int], list[list[int]]]:
        """
        Reads the JSON file at the given index and returns the encoded text
        and the list of citation indices contained in the file.
        """
        with open(os.path.join(TEXT_FP, "original", file_name + ".json"), "r") as f:
            opinion = json.load(f)
        return self.tokenizer.encode(opinion["txt"]), opinion["citation_indices"]

    def select_data_samples(
        self, encoded_text: torch.Tensor, vocab_indices: list[list[int]]
    ) -> tuple[list, list]:
        """
        Selects the data samples from the given encoded text.
        """

        context_windows: list[torch.Tensor] = []
        labels: list[int] = []

        if len(vocab_indices) <= 0:  # NOCIT
            labels.append(0)
            context_windows.append(self.select_random_sequence(encoded_text))
        else:
            offsets, citation_positions, encoded_text = self.get_offsets(encoded_text)

            for o in offsets:
                vidx = self.get_first_cit_idx(o, citation_positions)
                vkey = str(vocab_indices[vidx][0])
                if vkey in self.vocab.keys():
                    labels.append(int(self.vocab[vkey][0]))
                    context_windows.append(encoded_text[o - 256 : o])

        return context_windows, labels

    def select_random_sequence(self, encoded_text: torch.Tensor) -> torch.Tensor:
        """
        Selects a random sequence of length 256 from the given encoded text.
        """
        offset = np.random.randint(0, len(encoded_text) - 256)
        return encoded_text[offset : offset + 256]

    def pad(self, encoding: list[int]) -> torch.Tensor:
        """
        Pads the given encoding with the pad token ID to ensure the resulting tensor
        has a minimum length of context_size + forcasting_size.
        """
        padding_size = max(0, self.context_size + self.forcasting_size - len(encoding))
        padding = torch.tensor(
            [self.tokenizer.pad_token_id] * padding_size, dtype=torch.int16
        )
        return torch.cat([padding, torch.tensor(encoding, dtype=torch.int16)])

    def get_offsets(
        self, encoded_text: torch.Tensor
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        all_pos, possible_idx = self.possible_citation_positions(encoded_text)

        if len(possible_idx) <= 0:  # citation is in the first 256 tokens
            encoded_text = self.add_padding(encoded_text, all_pos)
            all_pos, possible_idx = self.possible_citation_positions(encoded_text)

        # sample offsets
        offsets: list[int] = []
        for i in possible_idx:
            possible_offsets = np.arange(
                start=all_pos[i] - 15,
                stop=all_pos[i] + 1,
            )
            offsets.append(np.random.choice(possible_offsets))

        return (offsets, all_pos, encoded_text)

    def possible_citation_positions(
        self, padded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the positions of all citation tokens in the padded tensor that meet
        the context and forcasting size constraints, along with the indices of those positions.
        """
        all_citation_pos = np.argwhere(padded == self.citation_token_id).ravel()
        possible_indices = np.argwhere(
            all_citation_pos > self.context_size + self.forcasting_size
        ).ravel()

        return all_citation_pos, possible_indices  # type: ignore

    def add_padding(
        self, encoded_text: torch.Tensor, all_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Adds additional padding to the beginning of the encoded text if the first citation
        token is not at least 256 tokens away from the beginning of the text.
        """
        additional_padding_size: int = (
            self.context_size + self.forcasting_size + 1 - int(all_pos[-1])
        )
        additional_padding = torch.tensor(
            [self.tokenizer.pad_token_id] * additional_padding_size,
            dtype=torch.int16,
        )
        return torch.cat([additional_padding, encoded_text])

    def get_first_cit_idx(self, offset: int, citation_positions: torch.Tensor) -> int:
        """
        Returns the index of the first citation token that occurs at or after the given offset.
        """
        indices = np.argwhere(citation_positions >= offset)
        return int(indices[0][0])

    def log_progress(self, fidx: int, fname: str) -> None:
        info: str = f"vocab_size_{len(vocab)}: {fidx}/{len(self.file_names)}: {fname}"
        if fidx % 50 == 0:
            logging.info(info)
        if fidx % 500 == 0:
            print(info)


if __name__ == "__main__":
    # parse CLI arguments ------------------------------------------------
    parser = argparse.ArgumentParser(description="Offline Preprocessing")
    parser.add_argument(
        "-s",
        "--start_at",
        required=False,
        default=0,
        type=int,
        help="Index to resume preprocessing at (in case it did not finish)",
    )
    args = vars(parser.parse_args())
    # --------------------------------------------------------------------

    for vsize in VOCAB_SIZES:
        vocab = load_vocab(vsize)

        tokenizer, _, cit_id = init_tokenizer()

        p = OfflinePreprocessor(
            tokenizer=tokenizer,
            citation_token_id=cit_id,
            vocab=vocab,
        )

        p.run(args["start_at"])