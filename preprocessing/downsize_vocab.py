import os
import pickle
from collections import OrderedDict
import json
import logging

from preprocessing.dataset_vocab import CitationVocabulary
from config import (
    VOCAB_FP,
    APPROXIMATE_VOCAB_SIZES,
    LOGS_FP,
    SortingStructure,
    NewVocabStructure,
)


logging.basicConfig(
    filename=os.path.join(LOGS_FP, "downsize_vocab.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)


def restructure_for_sorting(v: CitationVocabulary) -> SortingStructure:
    """Restructure the vocabulary for convenient sorting by citation occurance."""
    return [
        (count, string, old_index)
        for old_index, (string, count) in enumerate(v.citation_counts.items())
    ]


def seperate_tokens(
    v: SortingStructure,
) -> tuple[SortingStructure, SortingStructure]:
    """NOCIT, UNKCIT tokens are seperated from the rest of the vocabulary before sorting."""
    return v[:2], v[2:]


def sort_by_citation_occurance(v: SortingStructure) -> SortingStructure:
    """Sort the vocabulary by citation occurance."""
    return sorted(v, key=lambda l: l[0])


def downsize(v: SortingStructure) -> dict[int, NewVocabStructure]:
    """Downsize the vocabulary to the given sizes."""
    new_vocabs = dict()
    for vsize in APPROXIMATE_VOCAB_SIZES:
        step_size: int = len(original_vocab) // vsize
        downsized: SortingStructure = v[::step_size]
        new_vocab = restructure_vocab(downsized)
        new_vocabs[vsize] = new_vocab

    return new_vocabs


def restructure_vocab(v: SortingStructure) -> NewVocabStructure:
    """Restructure the vocabulary for convenient storage and checking of original indices."""
    return OrderedDict(
        {
            old_index: (new_index, string, count)
            for new_index, (count, string, old_index) in enumerate(v, start=2)
        }
    )


def reinsert_tokens(
    vs: dict[int, NewVocabStructure], nocit_unkit: SortingStructure
) -> dict[int, NewVocabStructure]:
    """Reinsert the NOCIT, UNKCIT tokens into the downsized vocabularies."""
    for i, (count, string, old_index) in zip([1, 0], nocit_unkit[::-1]):
        for vsize in APPROXIMATE_VOCAB_SIZES:
            vs[vsize][old_index] = (i, string, count)
            vs[vsize].move_to_end(old_index, last=False)
    return vs


def store_new_vocabs(vs: dict[int, NewVocabStructure]):
    """Store the downsized vocabularies in JSON files."""
    for vsize, v in vs.items():
        # actual size differs by 2-3 because of floor division and reinsertions
        vsize = len(v)
        with open(os.path.join(VOCAB_FP, f"size_{vsize}.json"), "w") as f:
            json.dump(v, f)
            logging.info(f"Stored vocab of size {vsize}.")


if __name__ == "__main__":
    # Load original vocabulary
    with open(os.path.join(VOCAB_FP, "original.pkl"), "rb") as f:
        original_vocab: CitationVocabulary = pickle.load(f)

    # Create smaller vocabularies
    restructured = restructure_for_sorting(original_vocab)
    nocit_unkcit, to_be_downsized = seperate_tokens(restructured)
    to_be_downsized = sort_by_citation_occurance(to_be_downsized)
    new_vocabs = downsize(to_be_downsized)
    new_vocabs = reinsert_tokens(new_vocabs, nocit_unkcit)
    store_new_vocabs(new_vocabs)
