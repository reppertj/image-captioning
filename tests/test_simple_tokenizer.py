import os

import pytest
from project.simple_tokenizer import WordTokenizer


@pytest.fixture
def trained_word_tokenizer(tmpdir):
    input = """this is a sentence\nthis is but another sentence\nthis is a third sentence
    \nthis is really long truncation not enabled so will it perfectly convert back
    """
    d = tmpdir.mkdir("test_tokenizer")
    filename = os.path.join(d, "input.txt")
    with open(filename, "w") as f:
        f.writelines(input)
    wt = WordTokenizer()
    wt.train(filename, 30, 1, ["[pad]", "[bos]", "[eos]", "[unk]"], None, None)
    return wt


def test_simple_tokenizer(trained_word_tokenizer):
    wt = trained_word_tokenizer

    assert wt.token_to_id("[pad]") == 0
    assert wt.token_to_id("[bos]") == 1
    assert wt.token_to_id("[eos]") == 2
    assert wt.token_to_id("[unk]") == 3

    assert wt.vocab_size == 22

    test_strs = [
        "this is a sentence",
        "this another sentence is",
        "this is really really long but truncation is not enabled so it will convert back perfectly",
        "sentence is",
    ]
    encoded = wt.encode_batch(test_strs)
    decoded = wt.decode_batch(encoded)
    assert decoded == test_strs


def test_simple_tokenizer_truncation(trained_word_tokenizer):
    wt = trained_word_tokenizer
    wt.enable_truncation(max_length=25)

    encoded = wt.encode_batch(["perfectly " * 50])

    assert len(encoded[0]) == 25


def test_simple_tokenizer_padding(trained_word_tokenizer):
    wt = trained_word_tokenizer
    wt.enable_truncation(max_length=10)
    wt.enable_padding()

    encoded = wt.encode_batch(["is " * 3, "will " * 10, "unknown " * 30])

    for element in encoded:
        assert len(element) == 10

    assert encoded[0][3] == 0
    assert encoded[1][-1] != 0
    assert encoded[2][-1] == 3


def test_simple_tokenizer_save_load(tmpdir, trained_word_tokenizer):
    wt = trained_word_tokenizer
    d = tmpdir.mkdir("test_save_tokenizer")

    wt.enable_truncation(30)
    wt.enable_padding()
    wt.save_model(d, "saved_tokenizer.pkl")

    wt2 = WordTokenizer()
    wt2.load_model(d, "saved_tokenizer.pkl")

    assert wt.tokens == wt2.tokens
    assert wt.padding == wt2.padding
    assert wt.vocab_size == wt2.vocab_size

    encoded = wt.encode_batch(["sample sentence to encode"])
    encoded2 = wt2.encode_batch(["sample sentence to encode"])
    assert encoded == encoded2
    
def test_simple_tokenizer_from_config(trained_word_tokenizer):
    wt = trained_word_tokenizer
    wt.enable_truncation(10)
    wt.enable_padding()
    
    wt_config = wt.config
    
    wt2 = WordTokenizer()
    wt2.load_config(wt_config)
    
    assert wt.tokens == wt2.tokens
    assert wt.padding == wt2.padding
    assert wt.vocab_size == wt2.vocab_size