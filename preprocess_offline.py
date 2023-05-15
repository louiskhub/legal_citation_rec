from __future__ import annotations

from typing import Tuple, List, Dict
import os
import json
import logging
import argparse

import numpy as np
import torch
from transformers import DebertaTokenizerFast

from config import OPINIONS_FP, ICLOUD_PATH


logging.basicConfig(
    filename=os.path.join(ICLOUD_PATH, "data_logs.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)


class OfflinePreprocessor:
    def __init__(
        self,
        opinions_dir: str,
        tokenizer: DebertaTokenizerFast,
        citation_token_id: int,
        set_type: str,  # "train", "dev", "test"
        context_size: int = 256,
        forcasting_size: int = 16,
    ) -> None:
        self.opinions_dir = opinions_dir
        self.file_names = self.get_filenames(set_type)
        self.tokenizer = tokenizer
        self.citation_token_id = citation_token_id
        self.context_size = context_size
        self.forcasting_size = forcasting_size

    def get_filenames(self, set_type: str) -> List[str]:
        """
        Returns the filenames of the given type.
        """
        with open(f"utils/data_split/{set_type}.txt", "r") as f:
            return [line.strip() for line in f.readlines() if line.strip() != ""]

    def run(self, start_at: int):
        for fi, fname in enumerate(self.file_names[start_at:], start=start_at):
            opinion_text, vocab_indices = self.load_data_from_disk(fname)
            encoding = self.tokenizer.encode(opinion_text)
            del opinion_text
            encoding = self.pad(encoding)
            try:
                offsets, citation_positions, encoding = self.get_offsets(encoding)
            except ValueError:
                continue

            context_windows: torch.Tensor = torch.empty(
                size=(len(offsets), 257), dtype=torch.int
            )
            for i, o in enumerate(offsets):
                citation_vocab_idx = self.get_first_cit_idx(o, citation_positions)
                label = torch.tensor(
                    [vocab_indices[citation_vocab_idx][0]], dtype=torch.int
                )
                context_windows[i] = torch.cat((encoding[o - 256 : o], label))

            torch.save(
                context_windows, os.path.join(ICLOUD_PATH, "data", fname + ".pt")
            )
            if fi % 50 == 0:
                logging.info(f"{fi}/{len(self.file_names)}: {fname}")
            if fi % 100 == 0:
                print(f"{fi}/{len(self.file_names)}: {fname}")

    def load_data_from_disk(self, file_name: str) -> Tuple[str, List[int]]:
        """
        Reads the opinion JSON file at the given index and returns the opinion text
        and the list of citation indices.
        """
        with open(os.path.join(self.opinions_dir, file_name + ".json"), "r") as f:
            opinion: Dict = json.load(f)
        return opinion["txt"], opinion["citation_indices"]

    def pad(self, encoding: List[int]) -> torch.Tensor:
        """
        Pads the given encoding with the pad token ID to ensure the resulting tensor
        has a minimum length of context_size + forcasting_size.
        """
        padding_size = max(0, self.context_size + self.forcasting_size - len(encoding))
        padding = torch.tensor([self.tokenizer.pad_token_id] * padding_size)
        return torch.cat([padding, torch.tensor(encoding)])

    def get_offsets(
        self, padded: torch.Tensor
    ) -> Tuple[int, np.ndarray[int], torch.Tensor]:
        all_pos, possible_idx = self.possible_citation_pos(padded)
        if len(all_pos) <= 0:
            raise ValueError("No possible indices found.")
        elif len(possible_idx) <= 0:
            add_pad: int = self.context_size + self.forcasting_size + 1 - all_pos[-1]
            padded = torch.cat(
                [torch.tensor([self.tokenizer.pad_token_id] * add_pad), padded],
            )
            all_pos, possible_idx = self.possible_citation_pos(padded)
        offsets = list()
        for idx in possible_idx:
            possible_offsets = np.arange(
                start=all_pos[idx] - 15,
                stop=all_pos[idx] + 1,
            )
            offsets.append(np.random.choice(possible_offsets))
        return (offsets, all_pos, padded)

    def possible_citation_pos(
        self, padded: torch.Tensor
    ) -> Tuple[np.ndarray[int], np.ndarray[int]]:
        """
        Returns the positions of all citation tokens in the padded tensor that meet
        the context and forcasting size constraints, along with the indices of those positions.
        """
        all_citation_pos = np.argwhere(padded == self.citation_token_id).ravel()
        possible_indices = np.argwhere(
            all_citation_pos > self.context_size + self.forcasting_size
        ).ravel()

        return all_citation_pos, possible_indices

    def get_first_cit_idx(
        self, offset: int, citation_positions: np.ndarray[int]
    ) -> int:
        """
        Returns the index of the first citation token that occurs at or after the given offset.
        """
        return np.argwhere(citation_positions >= offset)[0][0]


def init_tokenizer() -> Tuple[DebertaTokenizerFast, int, int]:
    tokenizer: DebertaTokenizerFast = DebertaTokenizerFast.from_pretrained(
        "microsoft/deberta-base"
    )
    tokenizer.add_tokens(["@pb@", "@cit@"])  # paragraphs + citations
    pb_id, cit_id = tokenizer.convert_tokens_to_ids(["@pb@", "@cit@"])
    return tokenizer, pb_id, cit_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Preprossing")
    parser.add_argument("-s", "--start_at", required=False, default=0, type=int)
    args = vars(parser.parse_args())

    tokenizer, _, cit_id = init_tokenizer()

    for set_type in ("train", "dev", "test"):
        p = OfflinePreprocessor(
            opinions_dir=OPINIONS_FP,
            tokenizer=tokenizer,
            citation_token_id=cit_id,
            set_type=set_type,
        )

        p.run(args["start_at"])
