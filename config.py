import os
from typing import TypeAlias, OrderedDict


# filepaths
OUTPUTS_FP: str = "outputs"
LOGS_FP: str = "logs"
TEXT_FP: str = os.path.join("data", "text")
VOCAB_FP: str = os.path.join("data", "vocabs")
SPLIT_FP: str = os.path.join("data", "splits")
GCLOUD_FP: str = "gs://legal_citation_rec/"

# hyper parameters
BASE_DISTILBERT: str = "distilbert-base-uncased"
CONTEXT_SIZE: int = 256  # input sequence length
FORCASTING_SIZE: int = (
    16  # maximum distance between citation token and last context token
)
LR: float = 1e-4  # learning rate

# data splits
TRAIN_SPLIT: float = 0.72
DEV_SPLIT: float = 0.18
TEST_SPLIT: float = 0.1
APPROXIMATE_VOCAB_SIZES: tuple[int, int, int, int] = (
    1429,
    857,
    476,
    102,
)  # actual ones differs slightly from actual sizes (by 2-3)
DOWNSIZED_VOCAB_SIZES: tuple[int, int, int, int] = (
    1431,
    859,
    479,
    105,
)

# reproducability
SEED: int = 42

# Type aliases for readability
SortingStructure: TypeAlias = list[tuple[int, str, int]]
NewVocabStructure: TypeAlias = OrderedDict[int, tuple[int, str, int]]
