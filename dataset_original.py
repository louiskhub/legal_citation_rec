import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DebertaTokenizerFast

#############################################
# Data sampling like in:
# Zihan Huang, Charles Low, Mengqiu Teng, Hongyi Zhang, Daniel E. Ho, Mark Krass and Matthias Grabmair,
# Context-Aware Legal Citation Recommendation using Deep Learning,
# Proceedings ICAIL 2021
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
        set_type: str,  # "train", "dev", "test"
        context_size: int = 256,
        forcasting_size: int = 16,
    ) -> None:
        """
        Initializes the dataset with the given directory containing opinion JSON files, tokenizer,
        citation token ID, context size, and forcasting size.
        """
        self.opinions_dir = opinions_dir
        self.file_names = self.get_filenames(set_type)
        self.tokenizer = tokenizer
        self.citation_token_id = citation_token_id
        self.context_size = context_size
        self.forcasting_size = forcasting_size

    def get_filenames(self, set_type: str) -> list[str]:
        """
        Returns the filenames of the given type.
        """
        with open(f"utils/data_split/{set_type}.txt", "r") as f:
            return [line.strip() for line in f.readlines() if line.strip() != ""]

    def __len__(self) -> int:
        """Returns the total number of opinion files in the dataset."""
        return len(self.file_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a random citation context window and the corresponding citation index
        from the opinion file at the given index.
        """
        opinion_text, vocab_indices = self.load_data_from_disk(idx)
        encoding = self.tokenizer.encode(opinion_text)
        del opinion_text
        encoding = self.pad(encoding)
        offset, citation_positions, encoding = self.get_offset(encoding)
        context_window: torch.Tensor = encoding[offset - 256 : offset]
        del encoding
        citation_vocab_idx = self.get_first_cit_idx(offset, citation_positions)
        label = torch.tensor(
            [vocab_indices[citation_vocab_idx][0]]  # type: ignore
        )  # BIAS (see paper)

        return context_window.int(), label

    def load_data_from_disk(self, idx: int) -> tuple[str, list[int]]:
        """
        Reads the opinion JSON file at the given index and returns the opinion text
        and the list of citation indices.
        """
        with open(os.path.join(self.opinions_dir, self.file_names[idx]), "r") as f:
            opinion: dict = json.load(f)

        # print(self.file_names[idx])  # Just for testing atm
        return opinion["txt"], opinion["citation_indices"]

    def pad(self, encoding: list[int]) -> torch.Tensor:
        """
        Pads the given encoding with the pad token ID to ensure the resulting tensor
        has a minimum length of context_size + forcasting_size.
        """
        padding_size = max(0, self.context_size + self.forcasting_size - len(encoding))
        padding = torch.tensor([self.tokenizer.pad_token_id] * padding_size)
        return torch.cat([padding, torch.tensor(encoding)])

    def get_offset(self, padded: torch.Tensor) -> tuple[int, np.ndarray, torch.Tensor]:
        """
        Returns the random offset for the context window, the citation positions,
        and the possibly updated padded tensor.
        """
        all_pos, possible_idx = self.possible_citation_pos(padded)
        if len(possible_idx) <= 0:
            add_pad: int = self.context_size + self.forcasting_size + 1 - all_pos[-1]
            padded = torch.cat(
                [torch.tensor([self.tokenizer.pad_token_id] * add_pad), padded],
            )
            all_pos, possible_idx = self.possible_citation_pos(padded)
        if len(possible_idx) == 0:
            print(self.tokenizer.decode(padded.int()))
        selected_idx: int = np.random.choice(possible_idx)
        possible_offsets = np.arange(
            start=all_pos[selected_idx] - 15,
            stop=all_pos[selected_idx] + 1,
        )
        offset: int = np.random.choice(possible_offsets)
        return (offset, all_pos, padded)

    def possible_citation_pos(
        self, padded: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the positions of all citation tokens in the padded tensor that meet
        the context and forcasting size constraints, along with the indices of those positions.
        """
        all_citation_pos = np.argwhere(padded == self.citation_token_id).ravel()
        possible_indices = np.argwhere(
            all_citation_pos > self.context_size + self.forcasting_size
        ).ravel()

        return all_citation_pos, possible_indices

    def get_first_cit_idx(self, offset: int, citation_positions: np.ndarray) -> int:
        """
        Returns the index of the first citation token that occurs at or after the given offset.
        """
        return np.argwhere(citation_positions >= offset)[0][0]
