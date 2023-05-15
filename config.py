import os

CONTEXT_SIZE: int = 256
FORCASTING_SIZE: int = 16
OPINIONS_FP: str = os.path.join("..", "preprocessed-cached-v4")
VOCAB_FP: str = os.path.join("..", "Vocab_min_20.pkl")
ICLOUD_FP: str = "../../Library/Mobile Documents/com~apple~CloudDocs/nlp"
OUTPUTS_FP: str = "outputs"
TRAIN_SPLIT: float = 0.72
DEV_SPLIT: float = 0.18
TEST_SPLIT: float = 0.1
SEED: int = 42
