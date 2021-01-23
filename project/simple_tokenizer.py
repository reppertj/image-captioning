import os
import pickle
from collections import Counter
from itertools import count
from pathlib import Path
from typing import List, Union


class WordTokenizer:
    """Implements a simple, pure-python, word-based tokenizer, with some of the
    same API as tokenizers.BertWordPieceTokenizer."""

    def __init__(self) -> None:
        self._incrementer = count()
        self.tokens = dict()
        self.padding = False
        self.truncation = False
        self.is_trained = False

    def check_trained(self):
        if not self.is_trained:
            raise ValueError("Train tokenizer before using it")

    def train(
        self,
        filename: Union[str, Path],
        vocab_size: int,
        min_frequency: int,
        special_tokens: List[int],
        initial_alphabet: str = None,
        limit_alphabet: int = None,
    ):
        """Train the word-based tokenizer from a file.

        Arguments:
            filename {Union[str, Path]} -- Path to training data with strings separated by newlines,
            with words split on whitespace. Does not strip punctuation or change case.
            vocab_size {int} -- Use at most this many words
            min_frequency {int} -- Only include words used at least this many times
            special_tokens {List[int]} -- Special tokens in order of PAD, BOS, EOS, UNK

        Keyword Arguments:
            initial_alphabet {str} -- Ignored; in for API compatibility (default: {None})
            limit_alphabet {int} -- Ignored; in for API compatibility (default: {None})
        """
        if self.is_trained:
            raise ValueError("Don't retrain; load config or use a new instance instead")
        self.max_vocab_size = vocab_size
        self.min_frequency = min_frequency

        if len(special_tokens) != 4:
            raise ValueError(
                "Special tokens must have exactly 4 values: PAD, BOS, EOS, UNK"
            )

        self.special_tokens = special_tokens

        for t in self.special_tokens:
            self.tokens[t] = next(self._incrementer)

        with open(filename, "r") as f:
            data = f.readlines()
        data = map(lambda s: s.lower().split(), data)
        counter = Counter()
        list(map(lambda lst: counter.update(lst), data))
        vocab = counter.most_common(self.max_vocab_size - len(self.tokens))
        vocab = filter(lambda tup: tup[1] >= self.min_frequency, vocab)
        vocab = filter(lambda tup: tup[0] not in self.tokens, vocab)
        for word, _ in vocab:
            self.tokens[word] = next(self._incrementer)
        self.ids = {v: k for k, v in self.tokens.items()}
        self.is_trained = True

    def token_to_id(self, token):
        self.check_trained()
        return self.tokens.get(token)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def get_vocab_size(self):
        return self.vocab_size

    def decode_batch(self, batch: List[List[int]], skip_special_tokens=True):
        return list(self._decode_batch(batch, skip_special_tokens))

    def _decode_batch(self, batch: List[List[int]], skip_special_tokens):
        self.check_trained()
        for element in batch:
            element = map(lambda i: self.ids.get(i, ""), element)
            if skip_special_tokens:
                element = filter(lambda w: w not in self.special_tokens, element)
            yield " ".join(element)

    def encode_batch(self, batch: List[str]):
        return list(self._encode_batch(batch))

    def _encode_batch(self, batch: List[str]):
        self.check_trained()
        for element in batch:
            split = element.strip().split(" ")
            if self.truncation and len(split) > self.max_length:
                split = split[: self.max_length]
            if self.padding and len(split) < self.max_length:
                split = split + [self.special_tokens[0]] * (
                    self.max_length - len(split)
                )
            yield list(map(lambda w: self.tokens.get(w, 3), split))

    def save_model(self, directory, filename):
        self.check_trained()
        with open(os.path.join(directory, filename), "wb") as f:
            pickle.dump(self.__dict__, f, 3)

    def load_model(self, directory, filename):
        with open(os.path.join(directory, filename), "rb") as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    def enable_padding(self):
        if not self.truncation:
            raise AttributeError("Enable truncation before enabling padding")
        self.padding = {"pad_id": 0}

    def enable_truncation(self, max_length):
        self.truncation = True
        self.max_length = max_length

    @property
    def config(self):
        return self.__dict__

    def load_config(self, config):
        self.__dict__.update(config)

