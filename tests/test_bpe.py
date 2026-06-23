"""Tests for the byte-level BPE tokenizer."""

import pytest

from bpe import BPETokenizer, merge, get_stats

CORPUS = (
    "the cat sat on the mat. the cat ran. a dog sat on a log. "
    "the quick brown fox jumps over the lazy dog. dogs and cats play. "
) * 20


@pytest.fixture
def tok():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=320)
    return t


def test_merge_replaces_pair():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_merge_no_match_is_identity():
    assert merge([1, 2, 3], (4, 5), 99) == [1, 2, 3]


def test_get_stats_counts_adjacent_pairs():
    stats = get_stats([1, 2, 1, 2, 3])
    assert stats[(1, 2)] == 2
    assert stats[(2, 3)] == 1


def test_base_bytes_always_present():
    # an untrained tokenizer still encodes anything via the 256 byte tokens
    t = BPETokenizer()
    t.train("", vocab_size=256)
    assert t.decode(t.encode("hello")) == "hello"


def test_roundtrip_ascii(tok):
    for s in ["the cat sat", "a dog", "quick brown fox", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_roundtrip_unicode(tok):
    # bytes-level BPE must survive arbitrary UTF-8, even unseen during training
    for s in ["café", "naïve façade", "emoji 🐱🚀", "日本語"]:
        assert tok.decode(tok.encode(s)) == s


def test_training_actually_merges(tok):
    # frequent words should compress below one-token-per-byte
    raw = len("the cat sat on the mat".encode("utf-8"))
    assert len(tok.encode("the cat sat on the mat")) < raw


def test_vocab_size_grows_with_specials(tok):
    before = tok.vocab_size
    tok.register_special(["<|endoftext|>", "<|user|>"])
    assert tok.vocab_size == before + 2


def test_special_tokens_spliced_as_single_id(tok):
    tok.register_special(["<|endoftext|>"])
    ids = tok.encode("hi<|endoftext|>bye")
    assert tok.special_tokens["<|endoftext|>"] in ids
    assert tok.decode(ids) == "hi<|endoftext|>bye"


def test_disable_special_treats_as_text(tok):
    tok.register_special(["<|endoftext|>"])
    ids = tok.encode("<|endoftext|>", allow_special=False)
    assert tok.special_tokens["<|endoftext|>"] not in ids


def test_save_load_roundtrip(tok, tmp_path):
    tok.register_special(["<|endoftext|>", "<|user|>"])
    path = tmp_path / "tok.json"
    tok.save(path)

    loaded = BPETokenizer().load(path)
    text = "the quick brown fox <|user|> hi"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
    assert loaded.vocab_size == tok.vocab_size
