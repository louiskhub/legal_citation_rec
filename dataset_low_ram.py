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
        self.file_names: list[str] = self.get_filenames(set_type)
        self.vocab_size: int = vocab_size

    def get_filenames(self, set_type: str) -> list[str]:
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample: dict[str, torch.Tensor] = torch.load(
            os.path.join(self.data_dir, self.file_names[idx])
        )
        random_i: int = random.randint(0, sample["inputs"].shape[0] - 1)
        context: torch.Tensor = sample["inputs"][random_i]
        label: torch.Tensor = sample["labels"][random_i]

        return context, label
