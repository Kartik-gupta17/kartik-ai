from ai.tokenizer import CharacterTokenizer


def test_fit():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("hello")

    assert tokenizer.vocabulary_size() == 8
    # PAD UNK BOS EOS h e l o


def test_encode():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("hello")

    ids = tokenizer.encode("hello")

    assert len(ids) == 5


def test_decode():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("hello")

    ids = tokenizer.encode("hello")

    text = tokenizer.decode(ids)

    assert text == "hello"


def test_unknown_character():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("abc")

    ids = tokenizer.encode("z")

    assert ids == [1]