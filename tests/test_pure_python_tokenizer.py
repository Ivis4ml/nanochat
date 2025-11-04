"""
Unit tests for the Pure Python BPE Tokenizer
"""

import tempfile
import os
from nanochat.pure_python_tokenizer import PurePythonBPETokenizer, train_tokenizer


def test_basic_training():
    """Test basic tokenizer training."""
    texts = ["hello world"] * 100
    tokenizer = train_tokenizer(iter(texts), vocab_size=300, verbose=False)

    assert tokenizer.get_vocab_size() > 256
    assert len(tokenizer.merges) > 0
    assert len(tokenizer.get_special_tokens()) == 9


def test_encode_decode():
    """Test encoding and decoding."""
    texts = ["hello world", "test test"] * 50
    tokenizer = train_tokenizer(iter(texts), vocab_size=300, verbose=False)

    test_cases = [
        "hello world",
        "test",
        "hello test world",
        "The quick brown fox",
    ]

    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Failed for: {repr(text)}"


def test_special_tokens():
    """Test special token handling."""
    texts = ["sample text"] * 50
    tokenizer = train_tokenizer(iter(texts), vocab_size=300, verbose=False)

    # Test all special tokens
    special_tokens = [
        "<|bos|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
        "<|python_start|>",
        "<|python_end|>",
        "<|output_start|>",
        "<|output_end|>",
    ]

    for token in special_tokens:
        token_id = tokenizer.encode_special(token)
        assert isinstance(token_id, int)
        assert token_id >= 256


def test_encode_with_bos():
    """Test encoding with BOS token."""
    texts = ["text"] * 50
    tokenizer = train_tokenizer(iter(texts), vocab_size=300, verbose=False)

    text = "hello"
    ids_without = tokenizer.encode(text, add_special_tokens=False)
    ids_with = tokenizer.encode(text, add_special_tokens=True)

    assert len(ids_with) == len(ids_without) + 1
    assert ids_with[0] == tokenizer.encode_special("<|bos|>")
    assert ids_with[1:] == ids_without


def test_save_load():
    """Test saving and loading tokenizer."""
    texts = ["save load test"] * 50
    tokenizer1 = train_tokenizer(iter(texts), vocab_size=300, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        tokenizer1.save(tmpdir)

        # Load
        tokenizer2 = PurePythonBPETokenizer.load(tmpdir)

        # Verify they produce same results
        test_text = "save load test"
        ids1 = tokenizer1.encode(test_text)
        ids2 = tokenizer2.encode(test_text)

        assert ids1 == ids2
        assert tokenizer1.get_vocab_size() == tokenizer2.get_vocab_size()


def test_unicode():
    """Test Unicode text handling."""
    texts = ["Hello 世界", "你好 world", "Привет мир"] * 50
    tokenizer = train_tokenizer(iter(texts), vocab_size=500, verbose=False)

    test_cases = [
        "Hello 世界",
        "你好 world",
        "Emoji: 🌍🎉",
        "Mixed: café résumé",
    ]

    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Unicode test failed for: {repr(text)}"


def test_empty_text():
    """Test encoding empty text."""
    texts = ["text"] * 50
    tokenizer = train_tokenizer(iter(texts), vocab_size=300, verbose=False)

    encoded = tokenizer.encode("")
    assert encoded == []

    decoded = tokenizer.decode([])
    assert decoded == ""


def test_vocab_size_limits():
    """Test vocabulary size constraints."""
    texts = ["test"] * 50

    # Should work with minimum size
    tokenizer = train_tokenizer(iter(texts), vocab_size=265, verbose=False)
    assert tokenizer.get_vocab_size() == 265

    # Should work with larger size
    tokenizer = train_tokenizer(iter(texts), vocab_size=1000, verbose=False)
    assert tokenizer.get_vocab_size() <= 1000  # May be less if not enough unique pairs


def test_compression():
    """Test that tokenizer achieves compression."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Python programming language.",
    ] * 100

    tokenizer = train_tokenizer(iter(texts), vocab_size=2048, verbose=False)

    test_text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(test_text)

    # Should achieve some compression
    num_bytes = len(test_text.encode('utf-8'))
    num_tokens = len(encoded)

    assert num_tokens < num_bytes, "Should achieve compression"


if __name__ == "__main__":
    print("Running Pure Python Tokenizer tests...")

    tests = [
        ("Basic training", test_basic_training),
        ("Encode/decode", test_encode_decode),
        ("Special tokens", test_special_tokens),
        ("Encode with BOS", test_encode_with_bos),
        ("Save/load", test_save_load),
        ("Unicode", test_unicode),
        ("Empty text", test_empty_text),
        ("Vocab size limits", test_vocab_size_limits),
        ("Compression", test_compression),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed! 🎉")
    else:
        exit(1)
