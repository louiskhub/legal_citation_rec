import os

import torch
from torch.utils.data import Dataset

##########################################################
# Data sampling from a single input file to reduce disk I/O
##########################################################


class CitationDataset(Dataset):
    """
    A PyTorch Dataset class for sampling citation context windows from a set of opinion documents.
    """

    def __init__(
        self,
        data_dir: str,
        set_type: str,  # "train", "dev", "test"
        vsize: int,
    ) -> None:
        self.fp: str = os.path.join(data_dir, f"vocab_size_{vsize}", set_type)
        self.inputs: torch.IntTensor = torch.load(os.path.join(self.fp, "inputs.pt"))
        self.labels: torch.IntTensor = torch.load(os.path.join(self.fp, "labels.pt"))

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]
