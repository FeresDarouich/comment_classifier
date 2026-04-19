from Tockenizer import WordTokenizer


def test_tokenizer_build_and_encode_length_and_cls():
    tok = WordTokenizer(vocab_size=50, max_len=8, lowercase=True, add_cls=True)
    tok.build_vocab(["Hello world", "Hello there"])

    ids = tok.encode("HELLO unknown")
    assert len(ids) == 8

    # CLS token should be first and non-pad
    assert ids[0] == tok.word2id[tok.CLS]


def test_attention_mask_padding():
    tok = WordTokenizer(vocab_size=20, max_len=6, lowercase=True, add_cls=False)
    tok.build_vocab(["a b c"])

    ids = tok.encode("a")
    mask = tok.attention_mask(ids)

    pad_id = tok.word2id[tok.PAD]
    for i, m in zip(ids, mask):
        assert m in (0, 1)
        assert (i == pad_id) == (m == 0)
