import os
import pickle
import json

from preprocessing.dataset_vocab import CitationVocabulary
from config import VOCAB_FP


if __name__ == "__main__":
    # Load original vocabulary
    with open(os.path.join(VOCAB_FP, "original.pkl"), "rb") as f:
        original_vocab: CitationVocabulary = pickle.load(f)

    new_vocab = {
        i: (i, k, v) for i, (k, v) in enumerate(original_vocab.citation_counts.items())
    }

    with open(os.path.join(VOCAB_FP, f"size_{len(original_vocab)}.json"), "w") as f:
        json.dump(new_vocab, f)
