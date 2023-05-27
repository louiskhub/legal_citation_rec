import os
import pickle
import json

from preprocessing.dataset_vocab import CitationVocabulary
from config import VOCAB_FP


if __name__ == "__main__":
    # Load original vocabulary
    with open(os.path.join(VOCAB_FP, "original.pkl"), "rb") as f:
        original_vocab: CitationVocabulary = pickle.load(f)

    # restructure the volcabulary to store it in the same format as the downsized ones
    new_vocab = {
        # original-index : (new-index, citation-string, citation-count)
        i: (i, k, v)
        for i, (k, v) in enumerate(original_vocab.citation_counts.items())
    }

    # save the vocabulary in the same directory as the downsized ones
    with open(os.path.join(VOCAB_FP, f"size_{len(original_vocab)}.json"), "w") as f:
        json.dump(new_vocab, f)
