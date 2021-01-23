import torch
import torch.nn.functional as F


# TODO: Try the function described in https://www.aclweb.org/anthology/2020.acl-main.93.pdf)
def temporal_softmax_loss(x, y, ignore_index):
    """ 
    x: tensor of (batch_size, d_1, ..., d_k, vocab_size)
    y: tensor of (batch_size, d_1, ..., d_k)
    ignore_index tells us which elements in the caption should not
    contribute to the loss (due to padding) """
    loss = F.cross_entropy(
        x.transpose(-1, 1), y, ignore_index=ignore_index, reduction="mean"
    )
    return loss


def multi_caption_temporal_softmax_loss(x, y, ignore_index):
    """
    x: Dict of "fcn" tensors of (batch_size, seq_length, vocab_size)
    y: Target tensor of (batch_size, n_fc_scores, seq_length)
    """
    loss = torch.zeros(1, device=x["fc0"].device)
    for i in range(y.shape[1]):
        scores = x["fc%d" % i].transpose(-1, 1)
        loss += F.cross_entropy(
            scores, y[:, i, :], ignore_index=ignore_index, reduction="mean"
        )
    return loss


def smoothing_temporal_softmax_loss(x, y, ignore_index, epsilon=0.1):
    """
    x: tensor of (batch_size, vocab_size, seq_length)
    y: tensor of (batch_size, seq_length)
    """
    num_classes = x.shape[1]
    log_preds = F.log_softmax(x, dim=1)
    loss = -log_preds.sum(dim=1).mean() / num_classes
    nll = F.nll_loss(log_preds, y, ignore_index=ignore_index, reduction="mean")
    return (epsilon * loss) + (1 - epsilon) * nll


def multi_caption_smoothing_temporal_softmax_loss(x, y, ignore_index, epsilon=0.1):
    """
    x: Dict of "fcn" tensors of (batch_size, seq_length, vocab_size)
    y: Target tensor of (batch_size, n_fc_scores, seq_length)
    """
    loss = torch.zeros(1, device=x["fc0"].device)
    for i in range(y.shape[1]):
        scores = x["fc%d" % i].transpose(-1, 1)
        loss += smoothing_temporal_softmax_loss(
            scores, y[:, i, :], ignore_index=ignore_index, epsilon=epsilon,
        )
    return loss
