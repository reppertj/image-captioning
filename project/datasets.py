"""
Classes and utility functions for loading image classification datasets
"""
import os
import json
from project.simple_tokenizer import WordTokenizer
from re import sub
from string import ascii_lowercase
from typing import List, Union

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from PIL import Image
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn

BOS = "[cls]"
EOS = "[sep]"
UNK = "[unk]"
PAD = "[pad]"


def load_coco_captions_json(json_path, img_dir):
    with open(json_path, "r") as f:
        annots = json.load(f)
    paths = annots["images"]
    captions = annots["annotations"]
    image_ids = {}
    for p in paths:
        image_ids[p["id"]] = [os.path.join(img_dir, p["file_name"])]
    for c in captions:
        if c["image_id"] not in image_ids:
            continue
        else:
            image_ids[c["image_id"]].append(c["caption"])
    captions_df = pd.DataFrame.from_dict(image_ids, orient="index")
    captions_df = (
        captions_df.iloc[:, :6].rename(columns={0: "path"}).reset_index(drop=True)
    )
    captions_df = captions_df.rename(
        columns={n: int(n) - 1 for n in captions_df.columns[1:]}
    )
    for col in captions_df.columns[1:]:
        captions_df[col] = (
            captions_df[col]
            .str.strip()
            .str.lower()
            .str.replace("'", "")
            .str.replace(r"[^a-z]", " ")
        )
    return captions_df


def load_flickr_csv(csv_path, img_dir):
    """
    Load the Flickr30k image dataset as available at
    https://www.kaggle.com/hsankesara/flickr-image-dataset
    and return a pandas dataframe with the file paths and captions.

    Args:
        csv_path (str): the path to the csv file containing the captions
        img_dir (str): the path to the directory containing the images
    """
    captions_df = pd.read_csv(csv_path, sep="|")
    captions_df.columns = captions_df.columns.str.lstrip()
    captions_df.comment_number[
        19999
    ] = " 4"  # The CSV is malformed on this line; fix manually
    captions_df["comment"][19999] = " A dog runs across the grass ."
    captions_df.comment_number = captions_df.comment_number.astype(np.uint8)
    captions_df.comment = (
        captions_df.comment.str.strip()
        .str.lower()
        .str.replace("'", "")
        .str.replace(r"[^a-z]", " ")
    )
    captions_df = captions_df.pivot(
        index="image_name", columns="comment_number"
    ).reset_index()
    captions_df = captions_df.set_axis([f"{y}" for _, y in captions_df.columns], axis=1)
    captions_df = captions_df.rename(columns={"": "path"})
    captions_df = captions_df.rename(
        columns={n: int(n) for n in captions_df.columns[1:]}
    )
    captions_df.path = captions_df.path.map(lambda p: os.path.join(img_dir, p))
    return captions_df


def remove_prefixes(captions_df: pd.DataFrame, prefixes: List[str]):
    """Strip a list of prefixes from captions (columns 1:end) of a
    pandas dataframe and return a copy.
    """
    for prefix in prefixes:
        captions_df.iloc[:, 1:] = captions_df.iloc[:, 1:].applymap(
            lambda s: sub("^\s*" + prefix, "", s)
        )
    return captions_df


def train_tokenizer_from_df(
    df,
    directory,
    filename,
    vocab_size,
    min_frequency,
    max_caption_length,
    special_tokens,
    use_bert_wordpiece=True,
):
    """
    Trains a tokenizer from a dataframe and saves to disk. Uses minimal alphabet
    of ascii lowercase plus up to 30 characters.

    Args:
        df: The dataframe containing the input strings. Skips the first column.
        directory: directory in which to save tokenizer files
        filename: filename for tokenizer model
        vocab_size: number of words to tokenizer
        min_frequency: required number of occurrences for a token
        special_tokens: list of special tokens for the model
    """
    if use_bert_wordpiece:
        tokenizer = BertWordPieceTokenizer(lowercase=True)
        tokenizer.enable_padding(length=max_caption_length, pad_id=0, pad_token=PAD)
        tokenizer.enable_truncation(
            max_length=max_caption_length, stride=0, strategy="longest_first"
        )
    else:
        tokenizer = WordTokenizer()
        tokenizer.enable_truncation(max_caption_length)
        tokenizer.enable_padding()
    strings = df.iloc[:, 1:].stack(-1).reset_index(drop=True)
    strings.to_csv(os.path.join(directory, filename), header=False, index=False)
    tokenizer.train(
        os.path.join(directory, filename),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ascii_lowercase,
        limit_alphabet=len(ascii_lowercase) + 30,
    )
    tokenizer.save_model(directory, filename + "tokenizer")
    return tokenizer


def load_pretrained_tokenizer(vocab_path, max_caption_length, special_tokens):
    pad_token, cls_token, sep_token, unk_token = special_tokens
    tokenizer = BertWordPieceTokenizer(
        vocab_path,
        pad_token=pad_token,
        unk_token=unk_token,
        sep_token=sep_token,
        cls_token=cls_token,
        lowercase=True,
    )
    tokenizer.enable_padding(length=max_caption_length, pad_id=0, pad_token=PAD)
    tokenizer.enable_truncation(
        max_length=max_caption_length, stride=0, strategy="longest_first"
    )
    return tokenizer


def add_special_tokens(df, pad=PAD, start=BOS, end=EOS, unk=UNK):
    """
    Add the start and end tokens to the strings in columns 1 -> end of a
    pandas dataframe. Returns a copy of the dataframe and a list of the
    special tokens.
    """
    for col in df.iloc[:, 1:].columns:
        if not df.loc[0, col].startswith(start):
            df[col] = start + " " + df[col] + " " + end
    return df, [pad, start, end, unk]


def tokens_to_ids(tokenizer, tokens):
    """
    Returns a dict of 'token: id' for tokens in a tokenizer.
    """
    if hasattr(tokenizer, "token_to_id"):
        return {t: tokenizer.token_to_id(t) for t in tokens}
    else:
        return {t: tokenizer.convert_tokens_to_ids(t) for t in tokens}


def vocab_size(tokenizer):
    return tokenizer.get_vocab_size()


def ids_to_captions(ids_tensor, tokenizer, skip_special_tokens=False):
    """
    Return a single captions or group of captions from a rank-1 or rank-2
    tensor of ids using a tokenizer.
    """
    if ids_tensor.dim() == 1:
        ids_tensor = ids_tensor.reshape(1, -1)
    ids_tensor = ids_tensor.cpu()
    strings = tokenizer.decode_batch(ids_tensor.tolist(), skip_special_tokens=False)
    if skip_special_tokens:
        strings = list(map(lambda s: s.lstrip(BOS).partition(EOS)[0], strings))
    return strings


def corpus_bleu_score(
    preds: torch.Tensor, gt: torch.Tensor, tokenizer, weights=(0.25, 0.25, 0.25, 0.25)
):
    """ Returns possibly weighted average of 1, 2, 3, and 4-gram corpus BLEU scores
    for a batch of predictions and ground truths. The tokenizer is
    used to strip special characters and padding and (if relevant)
    reconstruct words from sub-words.

    Arguments:
        preds {torch.Tensor} -- (N, seq_len)
        gt {torch.Tensor} -- (N, num_gt, seq_len)
    """
    preds = [s.strip().split(" ") for s in ids_to_captions(preds, tokenizer, True)]
    gt = [
        [s.strip().split(" ") for s in ids_to_captions(lst, tokenizer, True)]
        for lst in gt
    ]
    return corpus_bleu(gt, preds, weights=weights)


def sample_minibatch(minibatch, tokenizer, remove_special_tokens=True):
    """
    Sample a minibatch and show the images and captions.
    """
    inv_normalize = NormalizeInverse()
    sample_images = inv_normalize(minibatch["image"])
    sample_captions = minibatch["captions"]
    for i in range(sample_images.shape[0]):
        plt.imshow(sample_images[i].permute(1, 2, 0).clip(0, 1).cpu())
        plt.axis("off")
        caption_strs = ids_to_captions(
            sample_captions[i], tokenizer, remove_special_tokens
        )
        plt.title("\n".join(caption_strs))
        plt.show()


def visualize_tensor_image(image: torch.Tensor):
    """ (C, H, W) Does not undo normalization """
    with torch.no_grad():
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        for i in range(image.shape[0]):
            plt.imshow(image[i].permute(1, 2, 0).clip(0, 1).cpu())
            plt.axis("off")
            plt.show()


class NormalizeInverse(transforms.Normalize):
    """
    Invert an image normalization. Default values are for Flickr30k dataset.
    """

    def __init__(
        self, mean=(0.4435, 0.4201, 0.3837), std=(0.2814, 0.2734, 0.2820),
    ):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class CaptioningDataset(Dataset):
    """
    Pytorch dataset of Flickr and COCO images and captions. You probably do not want to
    create this class directly. Instead, use a CombinedDataModule to instantiate
    datasets.
    """

    def __init__(
        self,
        df,
        split,
        transform,
        target_transform,
        val_size,
        test_size,
        random_state=42,
    ):
        """ val_size, test_size can be int or float between 0 and 1. """
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.subset(df)

    def subset(self, df):
        if self.split not in {"train", "test", "val"}:
            raise ValueError
        train, test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )
        if self.split == "test":
            self.split_df = test
            return
        train, val = train_test_split(
            train, test_size=self.val_size, random_state=self.random_state
        )
        if self.split == "train":
            self.split_df = train
        elif self.split == "val":
            self.split_df = val

    def __len__(self):
        return self.split_df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = self.split_df.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB").copy()

        if self.transform:
            with torch.no_grad():
                image = self.transform(image)

        captions = self.split_df.iloc[idx, 1:].to_list()

        if self.target_transform:
            with torch.no_grad():
                captions = self.target_transform(captions)

        return {"image": image, "captions": captions}


class DatasetBuilder:
    """ Utility class to reduce boilerplate in data module """

    def __init__(
        self,
        captions_df,
        transform,
        target_transform,
        val_transform,
        val_target_transform,
        tokenizer,
        val_size,
        test_size,
    ):
        self.captions_df = captions_df
        self.transform = transform
        self.target_transform = target_transform
        self.val_transform = val_transform
        self.val_target_transform = val_target_transform
        self.tokenizer = tokenizer
        self.val_size = val_size
        self.test_size = test_size

    def new(self, split):
        if split in ("val", "test"):
            return CaptioningDataset(
                self.captions_df,
                split,
                self.val_transform,
                self.val_target_transform,
                self.val_size,
                self.test_size,
            )
        else:
            return CaptioningDataset(
                self.captions_df,
                split,
                self.transform,
                self.target_transform,
                self.val_size,
                self.test_size,
            )


class TokenizeTransform:
    __slots__ = ["tokenizer"]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, captions):
        """ A list of strings to a tensor """
        if isinstance(self.tokenizer, BertWordPieceTokenizer):
            return torch.tensor([t.ids for t in self.tokenizer.encode_batch(captions)])
        else:
            return torch.tensor([t for t in self.tokenizer.encode_batch(captions)])


class ShuffleCaptions:
    """ Shuffle a (n_captions, seq_len) tensor of captions """

    __slots__ = []

    def __call__(self, tensor):
        idxs = torch.randperm(tensor.shape[0])
        return tensor[idxs, :]


class CombinedDataModule(pl.LightningDataModule):
    """
    A data module for the Flickr30K and COCO datasets, available at:
    https://www.kaggle.com/hsankesara/flickr-image-dataset and
    https://cocodataset.org/#home

    The dataloaders return iterators of dicts of 'image' and 'captions',
    where 'image' is a (N, C, H, W) batch tensor of images and lists
    of captions, or, if transformed by the tokenizer (target_transform=`auto`),
    a (N, n_captions, vocab_size) batch tensor of tokenized
    captions.

    You must call `setup` first to make the dataloaders available.
    """

    def __init__(
        self,
        flickr_csv=None,
        flickr_dir=None,
        coco_json=None,
        coco_dir=None,
        pretrained_vocab: Union[None, os.PathLike] = None,
        use_bert_wordpiece=True,
        batch_size=64,
        val_size=1024,
        test_size=1024,
        remove_prefixes=True,
        transform="augment",  # 'normalize' to only normalize
        target_transform="shuffle",
        val_transform="normalize",
        val_target_transform="tokenize",
        vocab_size=5000,
        min_word_occurrences=1,
        max_caption_length=25,
        dev_set=None,  # int to limit datamodule size
        num_workers=4,
        pin_memory=True,
    ):
        super().__init__()
        self.flickr_csv = flickr_csv
        self.flickr_dir = flickr_dir
        self.coco_json = coco_json
        self.coco_dir = coco_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size

        self.remove_prefixes = (
            [
                "there is",
                "there are",
                "this is",
                "these are",
                "a photo of",
                "a picture of",
                "an image of",
            ]
            if remove_prefixes is True
            else remove_prefixes
        )

        self.transform = transform
        self.target_transform = target_transform
        self.val_transform = val_transform
        self.val_target_transform = val_target_transform

        self.pretrained_vocab = pretrained_vocab

        self.use_bert_wordpiece = use_bert_wordpiece

        self.vocab_size = vocab_size
        self.min_word_occurrences = min_word_occurrences
        self.max_caption_length = max_caption_length + 1  # including start token

        self.dev_set = dev_set

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.is_setup = False

    def setup(self, stage=None):
        if self.is_setup:
            """ Because we're shuffling the input, setup is not idempotent
            without this."""
            self.make_loader(stage)
            return None
        if self.flickr_csv is not None:
            captions_df_1 = load_flickr_csv(self.flickr_csv, self.flickr_dir)
        if self.coco_json is not None:
            captions_df_2 = load_coco_captions_json(self.coco_json, self.coco_dir)
        if self.flickr_csv is not None and self.coco_json is not None:
            captions_df = captions_df_1.append(
                captions_df_2, ignore_index=True
            ).reset_index(drop=True)
        elif self.flickr_csv is not None:
            captions_df = captions_df_1
        else:
            captions_df = captions_df_2
        idxs = np.array(captions_df.index)
        np.random.shuffle(idxs)
        captions_df = captions_df.iloc[idxs, :].reset_index(drop=True)

        if self.remove_prefixes:
            captions_df = remove_prefixes(captions_df, self.remove_prefixes)

        if self.dev_set:
            captions_df = captions_df.iloc[: self.dev_set]
        self.captions_df, self.special_tokens = add_special_tokens(captions_df)

        if not self.pretrained_vocab:
            self.tokenizer = train_tokenizer_from_df(
                self.captions_df,
                ".",
                "flickr30k_tokenizer",
                self.vocab_size,
                self.min_word_occurrences,
                self.max_caption_length,
                self.special_tokens,
                use_bert_wordpiece=self.use_bert_wordpiece,
            )
        else:
            self.tokenizer = load_pretrained_tokenizer(
                self.pretrained_vocab, self.max_caption_length, self.special_tokens
            )

        if self.transform == "augment":
            random_xforms = [
                transforms.RandomAffine(degrees=30, scale=(0.9, 1.1), shear=10),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
            ]
            img_xforms = [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(nn.ModuleList(random_xforms), p=0.5),
            ]
            self.transform = transforms.Compose(
                [
                    transforms.Compose(img_xforms),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.3),
                    transforms.Normalize(
                        (0.4435, 0.4201, 0.3837), (0.2814, 0.2734, 0.2820)
                    ),
                ]
            )
        elif self.transform == "normalize":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4435, 0.4201, 0.3837), (0.2814, 0.2734, 0.2820)
                    ),
                ]
            )

        if self.val_transform == "normalize":
            self.val_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4435, 0.4201, 0.3837), (0.2814, 0.2734, 0.2820)
                    ),
                ]
            )

        if self.target_transform == "shuffle":
            self.target_transform = transforms.Compose(
                [TokenizeTransform(self.tokenizer), ShuffleCaptions()]
            )
        elif self.target_transform == "tokenize":
            self.target_transform = transforms.Compose(
                [TokenizeTransform(self.tokenizer)]
            )

        if self.val_target_transform == "tokenize":
            self.val_target_transform = TokenizeTransform(self.tokenizer)

        self.dbuild = DatasetBuilder(
            self.captions_df,
            self.transform,
            self.target_transform,
            self.val_transform,
            self.val_target_transform,
            self.tokenizer,
            self.val_size,
            self.test_size,
        )
        self.make_loader(stage)
        self.is_setup = True

    def make_loader(self, stage):
        if stage == "fit" or stage is None:
            self.train = self.dbuild.new("train")
            self.val = self.dbuild.new("val")

        if stage == "test" or stage is None:
            self.test = self.dbuild.new("test")

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.test,
            batch_size=batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
