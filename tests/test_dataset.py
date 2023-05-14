import unittest
import os
import json
import random
import torch

from dataset_original import CitationDataset
from experimental.train import init_tokenizer
from config import CONTEXT_SIZE, FORCASTING_SIZE, OPINIONS_FP


class TestCitationDataset(unittest.TestCase):
    def setUp(self):
        self.file_names = [
            f for f in os.listdir(OPINIONS_FP) if f.lower().endswith(".json")
        ]
        self.random_file = random.choice(self.file_names)
        self.tokenizer, _, self.cit_id = init_tokenizer()
        self.dataset = CitationDataset(
            opinions_dir=OPINIONS_FP,
            tokenizer=self.tokenizer,
            citation_token_id=self.cit_id,
        )

    def test_load_data_from_disk(self):
        random_idx = self.file_names.index(self.random_file)
        opinion_text, vocab_indices = self.dataset.load_data_from_disk(random_idx)

        with open(os.path.join(self.dataset.opinions_dir, self.random_file), "r") as f:
            opinion_data = json.load(f)

        self.assertEqual(opinion_text, opinion_data["txt"])
        self.assertEqual(vocab_indices, opinion_data["citation_indices"])

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.file_names))

    def test_pad(self):
        opinion_text, _ = self.dataset.load_data_from_disk(
            self.file_names.index(self.random_file)
        )
        random_encoding = self.tokenizer.encode(opinion_text)
        padded = self.dataset.pad(random_encoding)
        self.assertGreaterEqual(len(padded), CONTEXT_SIZE + FORCASTING_SIZE)
        self.assertEqual(padded[-len(random_encoding) :].tolist(), random_encoding)

    def test_get_offset(self):
        opinion_text, _ = self.dataset.load_data_from_disk(
            self.file_names.index(self.random_file)
        )
        random_encoding = self.tokenizer.encode(opinion_text)
        padded = self.dataset.pad(random_encoding)
        offset, citation_positions, padded = self.dataset.get_offset(padded)
        self.assertGreaterEqual(offset, CONTEXT_SIZE)
        self.assertGreater(len(citation_positions), 0)

    def test_possible_citation_pos(self):
        example_padded = torch.tensor(
            [self.dataset.citation_token_id]
            * (self.dataset.context_size + self.dataset.forcasting_size)
        )
        all_citation_pos, possible_indices = self.dataset.possible_citation_pos(
            example_padded
        )
        self.assertEqual(len(all_citation_pos), len(example_padded))
        self.assertEqual(
            len(possible_indices),
            len(example_padded)
            - self.dataset.context_size
            - self.dataset.forcasting_size,
        )

    def test_get_first_cit_idx(self):
        example_citation_positions = torch.tensor([10, 20, 30, 40, 50])
        offset = 25
        first_cit_idx = self.dataset.get_first_cit_idx(
            offset, example_citation_positions
        )
        self.assertEqual(first_cit_idx, 2)

    # Add more unit tests for other methods here


if __name__ == "__main__":
    unittest.main()
