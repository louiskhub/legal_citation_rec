from __future__ import annotations

from typing import Tuple, List, Dict
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DebertaTokenizerFast

#############################################
# Data sampling like in the original paper
#############################################


class CitationDataset(Dataset):
    """
    A PyTorch Dataset class for sampling citation context windows from a set of opinion documents.
    """

    def __init__(
        self,
        opinions_dir: str,
        tokenizer: DebertaTokenizerFast,
        citation_token_id: int,
        context_size: int = 256,
        forcasting_size: int = 16,
    ) -> None:
        """
        Initializes the dataset with the given directory containing opinion JSON files, tokenizer,
        citation token ID, context size, and forcasting size.
        """
        self.opinions_dir = opinions_dir
        self.file_names = [
            f for f in os.listdir(opinions_dir) if f.lower().endswith(".json")
        ]
        self.tokenizer = tokenizer
        self.citation_token_id = citation_token_id
        self.context_size = context_size
        self.forcasting_size = forcasting_size

    def __len__(self) -> int:
        """Returns the total number of opinion files in the dataset."""
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a random citation context window and the corresponding citation index
        from the opinion file at the given index.
        """
        opinion_text, vocab_indices = self.load_data_from_disk(idx)
        encoding = self.tokenizer.encode(opinion_text)
        padded = self.pad(encoding)
        offset, citation_positions, padded = self.get_offset(padded)
        context_window = padded[offset - 256 : offset]
        citation_vocab_idx = self.get_first_cit_idx(offset, citation_positions)

        return context_window, vocab_indices[citation_vocab_idx][0]  # BIAS (see paper)

    def load_data_from_disk(self, idx: int) -> Tuple[str, List[int]]:
        """
        Reads the opinion JSON file at the given index and returns the opinion text
        and the list of citation indices.
        """
        with open(os.path.join(self.opinions_dir, self.file_names[idx]), "r") as f:
            opinion: Dict = json.load(f)

        # print(self.file_names[idx])  # Just for testing atm
        return opinion["txt"], opinion["citation_indices"]

    def pad(self, encoding: List[int]) -> torch.Tensor:
        """
        Pads the given encoding with the pad token ID to ensure the resulting tensor
        has a minimum length of context_size + forcasting_size.
        """
        padding_size = max(0, self.context_size + self.forcasting_size - len(encoding))
        padding = torch.tensor([self.tokenizer.pad_token_id] * padding_size)
        return torch.cat([padding, torch.tensor(encoding)])

    def get_offset(
        self, padded: torch.Tensor
    ) -> Tuple[int, np.ndarray[int], torch.Tensor]:
        """
        Returns the random offset for the context window, the citation positions,
        and the possibly updated padded tensor.
        """
        all_pos, possible_idx = self.possible_citation_pos(padded)
        if len(possible_idx) <= 0:
            add_pad: int = self.context_size - all_pos[-1]
            padded = torch.cat(
                [torch.tensor([self.tokenizer.pad_token_id] * add_pad), padded]
            )
            all_pos, possible_idx = self.possible_citation_pos(padded)

        selected_idx: int = np.random.choice(possible_idx)
        possible_offsets = np.arange(
            start=all_pos[selected_idx] - 15,
            stop=all_pos[selected_idx] + 1,
        )
        offset: int = np.random.choice(possible_offsets)
        return (offset, all_pos, padded)

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
