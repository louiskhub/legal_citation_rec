import os
import json
import logging
import argparse
from collections import OrderedDict

import numpy as np
import torch
from transformers import DistilBertTokenizerFast

from config import (
    LOGS_FP,
    TEXT_FP,
    CONTEXT_SIZE,
    FORCASTING_SIZE,
    DOWNSIZED_VOCAB_SIZES,
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
        self.fp = os.path.join(TEXT_FP, "preprocessed")

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
        Files are saved as .pt binary files in the data/text folder. The file names
        are the same as the original JSON files.
        """

        # preallocate tensors
        labels: torch.Tensor = torch.empty((0,), dtype=torch.int16)
        contexts: torch.Tensor = torch.empty((0, 256), dtype=torch.int16)

        n_saved: int = 0  # count of all the files saved

        for fidx, fname in enumerate(self.file_names[start_at:], start=start_at):
            encoded_text, vocab_indices = self.load_data_from_disk(fname)
            encoded_text = self.pad(encoded_text)
            current_contexts, current_labels = self.select_data_samples(
                encoded_text, vocab_indices
            )

            # only save if we could extract data based on the given vocab
            if current_labels:
                labels = torch.cat(
                    [labels, torch.tensor(current_labels, dtype=torch.int16)]
                )
                contexts = torch.vstack([contexts, torch.stack(current_contexts)])

            self.log_progress(fidx, fname)

            # save every 20k samples to disk since concatenating many tensors is more
            # expensive than concatenating a few large tensors later on
            # The size of 20k was chosen intuitively after monitoring the processing time
            if labels.shape[0] >= 20000:
                n_saved += 1
                # save current progress
                torch.save(
                    labels,
                    os.path.join(
                        self.fp, f"{n_saved}_labels_size_{len(self.vocab)}.pt"
                    ),
                )
                torch.save(
                    contexts,
                    os.path.join(
                        self.fp, f"{n_saved}_inputs_size_{len(self.vocab)}.pt"
                    ),
                )

                # reallocate tensor memory
                labels = torch.empty((0,), dtype=torch.int16)
                contexts = torch.empty((0, 256), dtype=torch.int16)

    def load_data_from_disk(self, file_name: str) -> tuple[list[int], list[list[int]]]:
        """
        Reads the JSON file at the given index and returns the encoded text
        and the list of citation indices contained in the file.
        """
        with open(os.path.join(TEXT_FP, "original", file_name + ".json"), "r") as f:
            opinion = json.load(f)

        # already encode the input sequence
        return self.tokenizer.encode(opinion["txt"]), opinion["citation_indices"]

    def select_data_samples(
        self, encoded_text: torch.Tensor, vocab_indices: list[list[int]]
    ) -> tuple[list, list]:
        """
        Selects the data samples from the given encoded text.
        """

        # preallocate lists
        context_windows: list[torch.Tensor] = []
        labels: list[int] = []

        if len(vocab_indices) <= 0:  # no citations in this opinion
            labels.append(0)  # NOCIT token has index 0
            # we can take any sequence from the text as input
            context_windows.append(self.select_random_sequence(encoded_text))

        else:
            offsets, citation_positions, encoded_text = self.get_offsets(encoded_text)

            for o in offsets:
                # from multiple citations, we only take the first one
                vidx = self.get_first_cit_idx(o, citation_positions)
                # convert into a key for the citation vocabulary
                vkey = str(vocab_indices[vidx][0])

                # the citation vocabulary has to include this index
                # this is important for the downsized vocabularies
                if vkey in self.vocab.keys():
                    new_label = int(self.vocab[vkey][0])
                    # we only take one label instance per opinion text
                    if new_label not in labels:
                        labels.append(new_label)
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
        # padding_size=0 if no padding is needed
        padding_size = max(0, self.context_size + self.forcasting_size - len(encoding))

        padding = torch.tensor(  # allocate padding tensor
            [self.tokenizer.pad_token_id] * padding_size, dtype=torch.int16
        )

        # include the padding at the beginning of the encoding
        return torch.cat([padding, torch.tensor(encoding, dtype=torch.int16)])

    def get_offsets(
        self, encoded_text: torch.Tensor
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """
        Returns the offsets of the context windows, the positions of all citations
        and the encoded text (sometimes with additional padding)
        """
        all_pos, possible_idx = self.possible_citation_positions(encoded_text)

        # citation is in the first 256 tokens and we cannot use it
        if len(possible_idx) <= 0:
            # additional padding is needed
            encoded_text = self.add_padding(encoded_text, all_pos)
            # recalculate possible citation positions
            all_pos, possible_idx = self.possible_citation_positions(encoded_text)

        # sample offsets
        offsets: list[int] = []
        for i in possible_idx:
            possible_offsets = np.arange(
                start=all_pos[i] - self.forcasting_size - 1,
                stop=all_pos[i] + 1,
            )
            offsets.append(np.random.choice(possible_offsets))

        return (offsets, all_pos, encoded_text)

    def possible_citation_positions(
        self, padded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the positions of all citation tokens in the given padded tensor and
        the citation vocabulary indices of the possible citation positions.
        """
        all_citation_pos = np.argwhere(padded == self.citation_token_id).ravel()

        # the minimum sequence length before a citation token is 256+16 in order to use it
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
        info: str = (
            f"vocab_size_{len(vocab)}: {fidx}/{len(self.file_names)}: {fname} processed"
        )
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

    # preprocess for original vocabulary and downsized vocabularies
    # Note: preprocessing only the original vocab size would be sufficient
    # and save a lot of time. However, during the time of writing this code,
    # it was not clear whether the original vocabulary size would be used.
    for vsize in (4287,) + DOWNSIZED_VOCAB_SIZES:
        vocab = load_vocab(vsize)

        tokenizer, _, cit_id = init_tokenizer()

        p = OfflinePreprocessor(
            tokenizer=tokenizer,
            citation_token_id=cit_id,
            vocab=vocab,
        )

        p.run(args["start_at"])
