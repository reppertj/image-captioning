from functools import reduce
import torch
from nltk.translate.bleu_score import corpus_bleu as _corpus_bleu, SmoothingFunction
from project.datasets import ids_to_captions
from pytorch_lightning.metrics import Metric

smoothie = SmoothingFunction().method4

def _corpus_bleu_score(
    preds: torch.Tensor, gts: torch.Tensor, tokenizer, weights=(0.25, 0.25, 0.25, 0.25)
):
    """ Returns (possibly weighted) average of 1, 2, 3, and 4-gram corpus BLEU scores
    for a batch of predictions and ground truths. The tokenizer is 
    used to strip special characters and padding and (if relevant)
    reconstruct words from sub-words.

    Arguments:
        preds {torch.Tensor} -- (N, seq_len)
        gt {torch.Tensor} -- (N, num_gt, seq_len)
        tokenizer {tokenizer}
        weights {sequence} -- weights for 1, 2, 3, 4-gram scores
    """
    total_preds = preds.shape[0]
    preds = [
        s.strip().split(" ")
        for s in ids_to_captions(preds, tokenizer, skip_special_tokens=True)
    ]
    gts = [
        [
            s.strip().split(" ")
            for s in ids_to_captions(lst, tokenizer, skip_special_tokens=True)
        ]
        for lst in gts
    ]
    new_preds, new_gts = [], []
    num_too_short = 0
    for i, pred in enumerate(preds):
        if len(pred) < 2:
            num_too_short += 1
        else:
            new_preds.append(pred)
            new_gts.append(gts[i])
    score = _corpus_bleu(new_gts, new_preds, weights=weights, smoothing_function=smoothie)
    return (score * (total_preds - num_too_short)) / total_preds


class CorpusBleu(Metric):
    """
    (Possibly weighted) average of 1, 2, 3, and 4-gram BLEU scores
    for a batch of predictions and ground truths. The tokenizer is 
    used to strip special characters and padding and (if relevant)
    reconstruct words from sub-words.
    
    init arguments:
        tokenizer {tokenizer}
        weights {sequence} -- weights for 1, 2, 3, 4-gram scores
        dist_sync_on_step -- Synchronize metric state across processes at each forward()
        before returning the value at the step.
    """

    def __init__(
        self, tokenizer, weights=(0.25, 0.25, 0.25, 0.25), dist_sync_on_step=False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

        self.tokenizer = tokenizer
        self.weights = weights

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state for single step; returns None. Call `forward` to 
        return a value, but note that BLEU score cannot be backpropagated.

        Arguments:
            preds {torch.Tensor} -- (N, seq_len_preds)
            target {torch.Tensor} -- (N, n_true_captions, seq_len_target)
        """
        assert preds.dim() == 2
        assert target.dim() == 3
        assert preds.shape[0] == target.shape[0]

        self.total_score += _corpus_bleu_score(
            preds, target, self.tokenizer, self.weights
        )
        self.n += 1

    def compute(self):
        return self.total_score / self.n.float()
