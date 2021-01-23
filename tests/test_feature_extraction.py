import pandas as pd
import torch
from project.datasets import train_tokenizer_from_df
from project.feature_extraction import (
    ImageFeatureExtractor,
    WordEmbedder,
)
from pytest import mark
from pytorch_lightning import seed_everything
from torch import nn

seed_everything(42)

if torch.cuda.is_available:
    device = "cpu"
else:
    device = "cpu"

df = pd.DataFrame(
    {
        "paths": ["path1", "path2", "path3"],
        0: [
            "the quick brown fox jumps over the lazy dog",
            "now is the time for all good folks",
            "everything is fine",
        ],
        1: [
            "what do you mean",
            "im sorry dave im afraid i cant do that",
            "yes this is a test",
        ],
        2: [
            "on the internet no one knows youre a dog",
            "i cant think of any more memes",
            "this one describes a picture",
        ],
    }
)

BOS = "[CLS]"
EOS = "[SEP]"
UNK = "[UNK]"
PAD = "[PAD]"

image_batch = torch.randn((16, 3, 224, 224), device=device)
caption_batch = torch.randint(1000, (16, 5, 30), device=device)


def test_image_extractor():
    imgs = image_batch.clone()
    ife_mobilenet = ImageFeatureExtractor().to(device)
    for param in ife_mobilenet.encoder.parameters():
        assert not param.requires_grad

    imgs = ife_mobilenet.encoder(imgs)
    assert imgs.shape == (16, 1280, 7, 7)

    assert isinstance(ife_mobilenet.pooling, nn.AdaptiveAvgPool2d)
    imgs = ife_mobilenet.pooling(imgs)
    assert imgs.shape == (16, 1280, 1, 1)

    assert isinstance(ife_mobilenet.projector, nn.Sequential)
    assert isinstance(list(ife_mobilenet.projector.children())[0], nn.Linear)
    imgs = imgs.view(16, -1)
    imgs = ife_mobilenet.projector(imgs)
    assert imgs.shape == (16, 128)

    assert not ife_mobilenet.convolution


def test_image_extractor_conv():
    imgs = image_batch.clone()
    ife_mobilenet = ImageFeatureExtractor(
        pooling=False, convolution_in="infer", projection_in=False,
    )
    for param in ife_mobilenet.encoder.parameters():
        assert not param.requires_grad

    imgs = ife_mobilenet.encoder(imgs)
    assert imgs.shape == (16, 1280, 7, 7)

    assert isinstance(ife_mobilenet.convolution, nn.Conv2d)
    imgs = ife_mobilenet.convolution(imgs)
    assert imgs.shape == (16, 128, 7, 7)

    assert not ife_mobilenet.pooling
    assert not ife_mobilenet.projector

