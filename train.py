from __future__ import annotations

from typing import Tuple
import pickle

from transformers import DebertaTokenizerFast, DebertaForSequenceClassification
from torch.utils.data import DataLoader
from dataset import CitationDataset
from config import VOCAB_FP, OPINIONS_FP


def load_vocab():
    with open(VOCAB_FP, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def init_model(embedding_len: int) -> DebertaForSequenceClassification:
    model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base")
    model.resize_token_embeddings(embedding_len)
    return model


def init_tokenizer() -> Tuple[DebertaTokenizerFast, int, int]:
    tokenizer: DebertaTokenizerFast = DebertaTokenizerFast.from_pretrained(
        "microsoft/deberta-base"
    )
    tokenizer.add_tokens(["@pb@", "@cit@"])  # paragraphs + citations
    pb_id, cit_id = tokenizer.convert_tokens_to_ids(["@pb@", "@cit@"])
    return tokenizer, pb_id, cit_id


if __name__ == "__main__":
    # v = load_vocab()

    tokenizer, _, cit_id = init_tokenizer()
    # model = init_model(len(tokenizer))

    test = CitationDataset(
        opinions_dir=OPINIONS_FP,
        tokenizer=tokenizer,
        citation_token_id=cit_id,
        set_type="train",
    )
    # loader = DataLoader(test, batch_size=1)
    # text, cit_idx = next(iter(loader))

    # print(tokenizer.decode(text[0].int()))
    # print(v.citation_str_by_index(cit_idx[0]))

    ex = test.__getitem__(0)
    print(ex)
