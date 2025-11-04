"""
Demo: Pure Python BPE Tokenizer

This script demonstrates how to use the pure Python BPE tokenizer implementation.
It shows:
1. Training a tokenizer from scratch
2. Encoding and decoding text
3. Saving and loading tokenizers
4. Comparing with the Rust implementation (if available)
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nanochat.pure_python_tokenizer import PurePythonBPETokenizer, train_tokenizer


def demo_basic_usage():
    """Demonstrate basic tokenizer usage."""
    print("=" * 80)
    print("DEMO 1: Basic Usage")
    print("=" * 80)

    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Python is a popular programming language for data science.",
        "Byte pair encoding is an efficient tokenization algorithm.",
        "GPT models use transformer architecture.",
        "Training neural networks requires large datasets.",
        "The tokenizer splits text into subword units.",
    ] * 200  # Repeat to have sufficient training data

    print(f"\nTraining data: {len(training_texts)} texts")
    print("Training tokenizer with vocab_size=1024...")

    # Train tokenizer
    t0 = time.time()
    tokenizer = train_tokenizer(iter(training_texts), vocab_size=1024, verbose=True)
    train_time = time.time() - t0

    print(f"\nTraining completed in {train_time:.2f} seconds")

    # Test encoding
    test_texts = [
        "Hello, world!",
        "The quick brown fox",
        "Machine learning is fascinating",
        "Python programming with transformers",
    ]

    print("\n" + "-" * 80)
    print("Encoding examples:")
    print("-" * 80)

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        matches = "✓" if text == decoded else "✗"

        print(f"\nText: {repr(text)}")
        print(f"Tokens: {len(encoded)}")
        print(f"IDs: {encoded}")
        print(f"Decoded: {repr(decoded)}")
        print(f"Exact match: {matches}")


def demo_detailed_visualization():
    """Show detailed token breakdown."""
    print("\n" + "=" * 80)
    print("DEMO 2: Detailed Visualization")
    print("=" * 80)

    # Quick training on simple data
    simple_texts = ["hello world", "hello there", "world peace"] * 100

    print("\nTraining small tokenizer...")
    tokenizer = train_tokenizer(iter(simple_texts), vocab_size=300, verbose=False)

    # Visualize encoding
    test_text = "hello world"
    print(f"\nVisualizing: {repr(test_text)}")
    tokenizer.visualize_encoding(test_text)


def demo_special_tokens():
    """Demonstrate special token usage."""
    print("\n" + "=" * 80)
    print("DEMO 3: Special Tokens")
    print("=" * 80)

    # Train tokenizer
    texts = ["This is a sample text."] * 100
    tokenizer = train_tokenizer(iter(texts), vocab_size=400, verbose=False)

    print("\nSpecial tokens in vocabulary:")
    for token_str in tokenizer.get_special_tokens():
        token_id = tokenizer.encode_special(token_str)
        print(f"  {token_str:25s} -> ID {token_id}")

    # Encode with BOS token
    text = "Hello, world!"
    ids_with_bos = tokenizer.encode(text, add_special_tokens=True)
    ids_without_bos = tokenizer.encode(text, add_special_tokens=False)

    print(f"\nText: {repr(text)}")
    print(f"Without BOS: {ids_without_bos}")
    print(f"With BOS:    {ids_with_bos}")


def demo_save_load():
    """Demonstrate saving and loading."""
    print("\n" + "=" * 80)
    print("DEMO 4: Save and Load")
    print("=" * 80)

    # Train tokenizer
    texts = [
        "Save and load demonstration.",
        "Tokenizers can be persisted to disk.",
        "This allows reusing trained tokenizers.",
    ] * 100

    print("\nTraining tokenizer...")
    tokenizer1 = train_tokenizer(iter(texts), vocab_size=400, verbose=False)

    # Save
    save_dir = "/tmp/tokenizer_demo"
    print(f"\nSaving to {save_dir}...")
    tokenizer1.save(save_dir)

    # Load
    print("\nLoading from disk...")
    tokenizer2 = PurePythonBPETokenizer.load(save_dir)

    # Verify they produce same results
    test_text = "Testing save and load functionality."
    ids1 = tokenizer1.encode(test_text)
    ids2 = tokenizer2.encode(test_text)

    print(f"\nTest text: {repr(test_text)}")
    print(f"Original tokenizer: {ids1}")
    print(f"Loaded tokenizer:   {ids2}")
    print(f"Match: {'✓' if ids1 == ids2 else '✗'}")


def demo_comparison():
    """Compare pure Python vs optimized versions (if available)."""
    print("\n" + "=" * 80)
    print("DEMO 5: Performance Comparison")
    print("=" * 80)

    # Prepare test data
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence.",
        "Natural language processing with transformers.",
    ] * 100

    training_texts = test_texts * 5

    # Train pure Python version
    print("\nTraining Pure Python tokenizer...")
    t0 = time.time()
    py_tokenizer = train_tokenizer(iter(training_texts), vocab_size=1024, verbose=False)
    py_train_time = time.time() - t0

    # Encode test texts
    print("Encoding test texts with Pure Python...")
    t0 = time.time()
    for text in test_texts:
        _ = py_tokenizer.encode(text)
    py_encode_time = time.time() - t0

    print(f"\nPure Python Results:")
    print(f"  Training time: {py_train_time:.3f}s")
    print(f"  Encoding time: {py_encode_time:.3f}s ({len(test_texts)} texts)")
    print(f"  Avg per text:  {py_encode_time / len(test_texts) * 1000:.3f}ms")

    # Try to compare with Rust version
    try:
        from nanochat.tokenizer import RustBPETokenizer

        print("\nTraining RustBPE tokenizer...")
        t0 = time.time()
        rust_tokenizer = RustBPETokenizer.train_from_iterator(iter(training_texts), 1024)
        rust_train_time = time.time() - t0

        print("Encoding test texts with RustBPE...")
        t0 = time.time()
        for text in test_texts:
            _ = rust_tokenizer.encode(text)
        rust_encode_time = time.time() - t0

        print(f"\nRustBPE Results:")
        print(f"  Training time: {rust_train_time:.3f}s")
        print(f"  Encoding time: {rust_encode_time:.3f}s ({len(test_texts)} texts)")
        print(f"  Avg per text:  {rust_encode_time / len(test_texts) * 1000:.3f}ms")

        print(f"\nSpeedup:")
        print(f"  Training: {py_train_time / rust_train_time:.1f}x faster with Rust")
        print(f"  Encoding: {py_encode_time / rust_encode_time:.1f}x faster with Rust")

    except ImportError:
        print("\nRustBPE not available - skipping comparison")
        print("(Pure Python version is educational; Rust version is for production)")


def demo_compression_ratio():
    """Analyze compression achieved by tokenizer."""
    print("\n" + "=" * 80)
    print("DEMO 6: Compression Analysis")
    print("=" * 80)

    # Train tokenizer
    texts = [
        "Compression ratio analysis for BPE tokenization.",
        "Tokenizers reduce the sequence length significantly.",
        "This is important for transformer efficiency.",
    ] * 200

    print("\nTraining tokenizer...")
    tokenizer = train_tokenizer(iter(texts), vocab_size=2048, verbose=False)

    # Test compression
    test_cases = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning with byte pair encoding.",
        "Natural language processing and tokenization algorithms.",
        "Transformers and attention mechanisms in neural networks.",
    ]

    print("\nCompression analysis:")
    print("-" * 80)
    print(f"{'Text':<50s} {'Bytes':>8s} {'Tokens':>8s} {'Ratio':>8s}")
    print("-" * 80)

    for text in test_cases:
        num_bytes = len(text.encode('utf-8'))
        tokens = tokenizer.encode(text)
        num_tokens = len(tokens)
        ratio = num_bytes / num_tokens

        print(f"{text[:47] + '...' if len(text) > 47 else text:<50s} "
              f"{num_bytes:8d} {num_tokens:8d} {ratio:8.2f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print(" " * 20 + "Pure Python BPE Tokenizer Demo")
    print("=" * 80)

    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Detailed Visualization", demo_detailed_visualization),
        ("Special Tokens", demo_special_tokens),
        ("Save and Load", demo_save_load),
        ("Performance Comparison", demo_comparison),
        ("Compression Analysis", demo_compression_ratio),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All demos complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
