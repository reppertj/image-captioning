import heapq
from dataclasses import dataclass
from itertools import islice
from typing import Any, List, Tuple, Union

import torch
import torch.nn.functional as F
import wandb
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from project.datasets import NormalizeInverse, ids_to_captions


inv_normalize = NormalizeInverse()


def sample_predictions(minibatch, model, n_captions="max", remove_special_tokens=False):
    if n_captions == "max":
        n_captions = model.decoder.num_rnns
    tokenizer = model.datamodule.tokenizer
    model.eval()
    with torch.no_grad():
        pred_captions = model.forward(minibatch, n_captions=n_captions)
        images = inv_normalize(minibatch["image"])
        ground_truth = minibatch["captions"]
        for i in range(ground_truth.shape[0]):
            plt.imshow(images[i].permute(1, 2, 0).clip(0, 1).cpu())
            plt.axis("off")
            gt_str = "\n".join(
                ids_to_captions(ground_truth[i], tokenizer, remove_special_tokens)
            )
            pred_str = "\n".join(
                ids_to_captions(pred_captions[i], tokenizer, remove_special_tokens)
            )
            plt.title(f"Ground truth: {gt_str}\nPrediction: {pred_str}")
            plt.show()


def log_wandb_preds(tokenizer, images, preds, ground_truth):
    """
    images: Tensor of (N, C, H, W)
    preds: Tensor of (N, M, T, L)
    ground_truth: Tensor of (N, M, T, L)
    Where N is batch_size, M is the number of captions, T is caption length,
    and L is words.

    Returns list of wandb.Image objects that can be sent to logger
    """
    to_PIL = ToPILImage()
    to_log = []
    for i in range(preds.shape[0]):
        image = to_PIL(inv_normalize(images[i].cpu()))
        gt_str = ", ".join(
            ids_to_captions(ground_truth[i], tokenizer, remove_special_tokens=True)
        )
        pred_str = ", ".join(
            ids_to_captions(preds[i], tokenizer, remove_special_tokens=True)
        )
        caption = f"TRUE: {gt_str}; PRED: {pred_str}"
        to_log.append(wandb.Image(image, caption=caption))
    return to_log


@dataclass
class Candidate:
    words_so_far: list
    yn: torch.tensor  # (wordvec_dim,)
    hn: Union[torch.Tensor, None]  # (hidden_dim,)
    cn: Union[torch.Tensor, None]  # (hidden_dim,)
    states: Union[tuple, None]


@dataclass
class PQCandidate:
    priority: float = 0.0
    candidate: Candidate = Any

    def __lt__(self, other):
        return self.priority < other.priority


def batch_beam_search(
    rnn_captioner,
    yns: torch.Tensor,
    hns: Union[torch.Tensor, None],
    cns: Union[torch.Tensor, None],
    states: Union[Tuple[torch.Tensor], None],
    features: Union[torch.Tensor, None],
    max_length: int,
    which_rnn: int = 0,
    alpha: float = 0.7,
    beam_width: int = 10,
):
    """[summary]
    yn: (batch_size, 1, wordvec_dim)
    hn, cn, state[0], state[1]: (batch_size, 1, hidden_size)
    features: (batch_size, num_features, pixels, pixels)
    """
    batch_size = yns.shape[0]

    captions = torch.full(
        (batch_size, max_length + 1),
        rnn_captioner._pad,
        device=yns.device,
        dtype=torch.long,
    )
    captions[:, 0] = rnn_captioner._start

    for i in range(batch_size):
        yn = yns[i : i + 1, :, :]
        hn = None if hns is None else hns[:, i : i + 1, :]  # (rnn_depth, 1, hidden_size)
        cn = None if cns is None else cns[:, i : i + 1, :]  # (rnn_depth, 1, hidden_size)
        state = (
            None
            if states is None
            else (states[0][:, i : i + 1, :], states[1][:, i : i + 1, :])
        )
        feature = None if features is None else features[i : i + 1, :, :, :]  # (1, num_features, pixels, pixels)
        best = single_beam_search(
            [
                PQCandidate(
                    candidate=Candidate(
                        words_so_far=[], yn=yn, hn=hn, cn=cn, states=state
                    )
                )
            ],
            rnn_captioner=rnn_captioner,
            features=feature,
            num_words_left=max_length,
            which_rnn=which_rnn,
            alpha=alpha,
            beam_width=beam_width,
        )
        captions[i, 1:] = torch.tensor(
            best.candidate.words_so_far, dtype=captions.dtype, device=captions.device
        )
    return captions


def single_beam_search(
    candidates: List[PQCandidate],
    rnn_captioner,
    features: Union[torch.Tensor, None],
    num_words_left: int,
    which_rnn: int = 0,
    alpha: float = 0.7,
    beam_width: int = 10,
):
    """
    TODO: Vectorize this fully loopy implementation of beam search
    by, e.g., expanding hidden tensors by the beam width dimension
    """
    if num_words_left == 0:
        return next(candidates)
    to_consider = []
    for candidate in candidates:
        to_consider = heapq.merge(
            to_consider,
            get_new_candidates(
                candidate, rnn_captioner, features, which_rnn, beam_width, alpha
            ),
        )
    candidates = islice(to_consider, beam_width)
    return single_beam_search(
        candidates=candidates,
        rnn_captioner=rnn_captioner,
        features=features,
        num_words_left=(num_words_left - 1),
        which_rnn=which_rnn,
        alpha=alpha,
        beam_width=beam_width,
    )


def get_new_candidates(
    pqcandidate: PQCandidate,
    rnn_captioner,
    features: Union[torch.Tensor, None] = None,
    which_rnn: int = 0,
    num_new: int = 1,
    alpha: float = 0.7,
):
    """Generate list of new candidates from a given candidate, with their priorities

    Arguments:
        candidate {Candidate} -- [description]
        rnn_captioner {CaptioningRNN} -- [description]

    Keyword Arguments:
        features {Union[torch.Tensor, None]} -- [description] (default: {None})
        which_rnn {int} -- [description] (default: {0})
        num_new {int} -- [description] (default: {1})
        alpha {float} -- Length normalization factor (default: {0.7})
    """
    candidate = pqcandidate.candidate
    old_priority = pqcandidate.priority
    assert candidate.yn.shape == (1, 1, rnn_captioner.word_embedder.wordvec_dim)
    if candidate.hn is not None:
        assert candidate.hn.shape == (
            rnn_captioner.num_rnn_layers * rnn_captioner.num_rnn_directions * 1,
            1,
            rnn_captioner.decoder.hidden_size,
        )
    if candidate.cn is not None:
        assert candidate.cn.shape == (1, rnn_captioner.decoder.hidden_size)
    if candidate.states is not None:
        assert candidate.states[0].shape == (1, 1, rnn_captioner.decoder.hidden_size)
        assert candidate.states[1].shape == (1, 1, rnn_captioner.decoder.hidden_size)
    if features is not None:
        assert features.shape[:2] == (1, rnn_captioner.decoder.num_features)
    if rnn_captioner.rnn_type in ("rnn", "gru"):
        output, hn = getattr(rnn_captioner.decoder, "rnn%d" % which_rnn)(
            candidate.yn, candidate.hn.contiguous()
        )
        cn, states = None, None
    elif rnn_captioner.rnn_type == "lstm":
        output, states = getattr(rnn_captioner.decoder, "rnn%d" % which_rnn)(
            candidate.yn, candidate.states
        )
        hn, cn = None, None
    else:  # rnn_captioner.rnn_type == 'attention'
        output, hn, cn = getattr(rnn_captioner.decoder, "rnn%d" % which_rnn)(
            candidate.yn, features, candidate.hn, candidate.cn
        )
        states = None
    scores = getattr(rnn_captioner.fc_scorer, "fc%d" % which_rnn)(output)
    topk_scores, idxs = scores.topk(k=num_new, dim=2)  # (1, 1, num_new)
    topk_scores = F.log_softmax(topk_scores, dim=2).squeeze()  # (num_new,)
    yns = rnn_captioner.word_embedder(idxs).squeeze()  # (num_new, wordvec_dim)
    idxs = idxs.squeeze()  # (num_new,)

    if num_new == 1:
        topk_scores = topk_scores.unsqueeze(0)
        idxs = idxs.unsqueeze(0)
        yns = yns.unsqueeze(0)

    ret = []
    norm_factor = 1.0 / (len(candidate.words_so_far) + 1e-7) * alpha
    for i in range(idxs.shape[0]):
        priority = (old_priority + (-1 * topk_scores[i])) / norm_factor
        heapq.heappush(
            ret,
            PQCandidate(
                priority=priority,
                candidate=Candidate(
                    words_so_far=(candidate.words_so_far + [idxs[i]]),
                    yn=yns[i].view(1, 1, -1),
                    hn=hn,
                    cn=cn,
                    states=states,
                ),
            ),
        )
    return ret

    # x: hidden_dim
    # scores: vocab_size
    # idxs: beam_width
    # winners:
    # output: hidden_size
    # beam_width

    # @dataclass
    # class Candidate:
    #     words_so_far: list
    #     yn: torch.tensor  # (wordvec_dim,)
    #     hn: Union[torch.Tensor, None]  # (hidden_dim,)
    #     cn: Union[torch.Tensor, None]  # (hidden_dim,)
    #     states: Union[tuple, None]

    return
