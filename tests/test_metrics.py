from collections import defaultdict
from itertools import count
from typing import List

import torch
from project.datasets import BOS, EOS, PAD, UNK
from project.metrics import _corpus_bleu_score


class DecodeOnlyTokenizer:
    def __init__(self):
        self.token_factory = (str(i) for i in count(start=10))
        self.tokens = defaultdict(lambda: next(self.token_factory))
        self.special_tokens = {0: PAD, 1: BOS, 2: EOS, 3: UNK}
        self.tokens[0], self.tokens[1], self.tokens[2], self.tokens[3] = map(
            lambda k: self.special_tokens[k], (0, 1, 2, 3)
        )

    def decode_batch(self, batch: List[List[int]], skip_special_tokens=True):
        for instance in batch:
            yield " ".join(
                [
                    self.tokens[i] if i not in self.special_tokens else ""
                    for i in instance
                ]
            )


def test_corpus_bleu():
    perfect_preds = torch.full((12, 1, 22), 12, dtype=torch.long)
    gt_for_perfect_preds = torch.full((12, 5, 22), 12, dtype=torch.long)

    tokenizer = DecodeOnlyTokenizer()

    perfect_score = _corpus_bleu_score(
        perfect_preds.squeeze(), gt_for_perfect_preds, tokenizer
    )

    assert perfect_score == 1.0

    wrong_preds = torch.full((12, 1, 22), 15, dtype=torch.long)
    gt_for_wrong_preds = torch.full((12, 5, 22), 31, dtype=torch.long)

    worst_score = _corpus_bleu_score(
        wrong_preds.squeeze(), gt_for_wrong_preds, tokenizer
    )

    assert worst_score == 0.0
