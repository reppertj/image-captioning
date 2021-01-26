import torch
import matplotlib.pyplot as plt
from project.datasets import NormalizeInverse


def attention_visuals(minibatch, model):
    images = minibatch["image"]
    preds, weights = model(minibatch, return_attn=True, n_captions=1)
    batch_size, C, H, W = images.shape
    ret = []
    for i in range(batch_size):
        pred = preds[i, :, 1:].squeeze()
        if 2 in pred:
            stop_idx = torch.nonzero(pred == 2).min().item()
        else:
            stop_idx = pred.shape[0] + 1
        pred = pred[: (stop_idx + 1)]
        weight = weights[i].squeeze()[: (stop_idx + 1)]
        pred = model.tokenizer.decode_batch(
            [[p] for p in pred.cpu().numpy()], skip_special_tokens=False
        )
        visual_info = {"image": images[i], "pred": pred, "weight": weight}
        ret.append(visual_info)
    return ret


def combine_words_weights(words, weights, start_idx, stop_idx):
    ret_wds = []
    length = len(words) - (stop_idx - start_idx - 1)
    ret_tensor = torch.empty(
        length, weights.shape[-1], weights.shape[-1], dtype=weights.dtype
    )
    for i, wd in enumerate(words):
        if i < start_idx:
            ret_wds.append(wd)
            ret_tensor[i] = weights[i]
        elif i == start_idx:
            combined = wd
            ret_tensor[i] = weights[i]
        elif i < stop_idx:
            combined += wd.strip("#")
            ret_tensor[len(ret_wds) - 1] = torch.sum(
                torch.stack([ret_tensor[len(ret_wds) - 1], weights[i]]), dim=0
            )
        else:
            if combined:
                ret_wds.append(combined)
                combined = ""
            ret_wds.append(wd)
            ret_tensor[len(ret_wds) - 2] = weights[i]
    ret_tensor[start_idx] / (stop_idx - start_idx + 1)
    return ret_wds, ret_tensor


def get_combined_idxs(words):
    i = 0
    ret = []
    start_idx, stop_idx = -1, -1
    while (i + 1) < len(words) and words[i + 1] != "[sep]":
        if words[i][0] != "#" and words[i + 1][0] == "#":
            start_idx = i
            stop_idx = i + 2
        elif words[i + 1][0] == "#" and start_idx > -1:
            stop_idx += 1
        elif start_idx != -1:
            ret.append((start_idx, stop_idx))
            start_idx, stop_idx = -1, -1
        i += 1
    if start_idx != -1:
        ret.append((start_idx, stop_idx))
    return ret


def combine_all_words_weights(words, weights):
    idxs = get_combined_idxs(words)
    while idxs:
        words, weights = combine_words_weights(words, weights, *idxs[0])
        idxs = get_combined_idxs(words)
    return words, weights


def visualize_minibatch_weights(minibatch, model):
    n_inv = NormalizeInverse()
    extent = 0, 223, 0, 223
    visual_info = attention_visuals(minibatch, model)
    for item in visual_info:
        image = n_inv(item["image"]).cpu().permute(1, 2, 0).clip(0, 1)
        pred = item["pred"]
        weight = item["weight"]
        pred, weight = combine_all_words_weights(pred, weight)
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(" ".join(pred))
        plt.axis("off")
        nrows, ncols = (len(pred) // 4) + (len(pred) % 4 > 0), 4
        _, axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 3))
        axs = axs.flatten().tolist()
        for i, wd in enumerate(pred):
            frame_weights = weight[i]
            plt.sca(axs[i])
            plt.imshow(image, extent=extent)
            plt.imshow(
                frame_weights.detach().numpy(),
                interpolation="bilinear",
                alpha=0.5,
                extent=extent,
            )
            plt.title(wd)
        for ax in axs:
            plt.sca(ax)
            plt.axis("off")
        plt.show()
