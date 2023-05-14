from __future__ import annotations

from typing import Tuple, List
import os
import random

import torch
from torch.utils.data import Dataset

##########################################################
# Data sampling with lower RAM (data already tokenized)
##########################################################


class CitationDataset(Dataset):
    """
    A PyTorch Dataset class for sampling citation context windows from a set of opinion documents.
    """

    def __init__(
        self,
        data_dir: str,
        set_type: str,
        vocab_size: int,  # "train", "dev", "test"
    ) -> None:
        self.data_dir: str = data_dir
        self.file_names: List[str] = self.get_filenames(set_type)
        self.vocab_size: int = vocab_size

    def get_filenames(self, set_type: str) -> List[str]:
        """
        Returns the filenames of the given type.
        """
        with open(f"utils/data_split/{set_type}.txt", "r") as f:
            return [
                line.strip() + ".pt" for line in f.readlines() if line.strip() != ""
            ]

    def __len__(self) -> int:
        """Returns the total number of opinion files in the dataset."""
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        citations: torch.Tensor = torch.load(
            os.path.join(self.data_dir, self.file_names[idx])
        )
        print(citations)
        sample: torch.Tensor = citations[random.randint(0, citations.shape[0] - 1)]
        print(sample)
        label: torch.Tensor = torch.zeros(self.vocab_size, dtype=torch.int)
        label[sample[-1]] = 1

        return sample[:-1], label
