"""
Pure Python implementation of BPE (Byte Pair Encoding) Tokenizer
Based on GPT-4 style tokenization, but written entirely in Python for educational purposes.

This implementation demonstrates the core BPE algorithm without dependencies on Rust or complex libraries.
It's slower than the optimized versions but much easier to understand and modify.

Key Features:
- GPT-4 style regex-based text splitting
- Byte-level BPE encoding
- Special token support
- Save/load functionality
- Clear, readable code for learning
"""

import json
import os
import pickle
import regex  # Note: requires `regex` package for Unicode support (pip install regex)
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Iterator


# Special tokens used in nanochat
SPECIAL_TOKENS = [
    "<|bos|>",           # Beginning of sequence
    "<|user_start|>",    # User message start
    "<|user_end|>",      # User message end
    "<|assistant_start|>", # Assistant message start
    "<|assistant_end|>",   # Assistant message end
    "<|python_start|>",    # Python code start
    "<|python_end|>",      # Python code end
    "<|output_start|>",    # Output start
    "<|output_end|>",      # Output end
]

# GPT-4 style splitting pattern (slightly modified for smaller vocab)
# NOTE: Uses \p{N}{1,2} instead of \p{N}{1,3} to save tokens on numbers
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class PurePythonBPETokenizer:
    """
    A pure Python implementation of Byte Pair Encoding (BPE) tokenizer.

    This is an educational implementation that demonstrates how BPE works
    without relying on Rust, tiktoken, or other optimized libraries.
    """

    def __init__(self):
        """Initialize an empty tokenizer."""
        self.merges: Dict[Tuple[int, int], int] = {}  # (token_a, token_b) -> merged_token_id
        self.vocab: Dict[int, bytes] = {}  # token_id -> token_bytes
        self.pattern = SPLIT_PATTERN
        self.compiled_pattern = regex.compile(self.pattern)
        self.special_tokens: Dict[str, int] = {}  # special_token_str -> token_id
        self.vocab_size = 0

    def train(self, text_iterator: Iterator[str], vocab_size: int, verbose: bool = True) -> None:
        """
        Train the BPE tokenizer on text data.

        Args:
            text_iterator: Iterator yielding text strings
            vocab_size: Target vocabulary size (must be >= 256 + num special tokens)
            verbose: Whether to print training progress
        """
        if verbose:
            print(f"Starting BPE training with target vocab_size={vocab_size}")

        # Step 1: Split text into chunks using regex pattern and count frequencies
        if verbose:
            print("Step 1: Splitting text and counting chunks...")

        chunk_counts = Counter()
        num_docs = 0

        for text in text_iterator:
            num_docs += 1
            if verbose and num_docs % 10000 == 0:
                print(f"  Processed {num_docs} documents, {len(chunk_counts)} unique chunks")

            # Split text using the regex pattern
            for match in self.compiled_pattern.finditer(text):
                chunk = match.group()
                chunk_counts[chunk] += 1

        if verbose:
            print(f"  Total documents: {num_docs}")
            print(f"  Unique chunks: {len(chunk_counts)}")

        # Step 2: Convert chunks to sequences of byte IDs
        if verbose:
            print("Step 2: Converting chunks to byte sequences...")

        # Each chunk becomes a list of byte values (0-255)
        chunk_byte_seqs = {}
        for chunk, count in chunk_counts.items():
            byte_seq = list(chunk.encode('utf-8'))
            chunk_byte_seqs[tuple(byte_seq)] = count

        # Step 3: Perform BPE merges
        if verbose:
            print(f"Step 3: Performing BPE merges (need {vocab_size - 256} merges)...")

        # Initialize vocabulary with single bytes (0-255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_token_id = 256

        # Reserve space for special tokens
        num_special = len(SPECIAL_TOKENS)
        target_merges = vocab_size - 256 - num_special

        if target_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + num_special}")

        # BPE training loop
        for merge_iter in range(target_merges):
            # Count all pairs in all chunks
            pair_counts = defaultdict(int)

            for byte_seq, count in chunk_byte_seqs.items():
                if len(byte_seq) < 2:
                    continue
                for i in range(len(byte_seq) - 1):
                    pair = (byte_seq[i], byte_seq[i + 1])
                    pair_counts[pair] += count

            if not pair_counts:
                if verbose:
                    print(f"  No more pairs to merge at iteration {merge_iter}")
                break

            # Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            # Record this merge
            self.merges[best_pair] = next_token_id

            # Build the bytes for this new token
            token_a, token_b = best_pair
            self.vocab[next_token_id] = self.vocab[token_a] + self.vocab[token_b]

            if verbose and (merge_iter + 1) % 100 == 0:
                progress = (merge_iter + 1) / target_merges * 100
                print(f"  Progress: {progress:.1f}% ({merge_iter + 1}/{target_merges}) - "
                      f"Merged {best_pair} -> {next_token_id} (count: {best_count})")

            # Update all chunks by applying this merge
            new_chunk_byte_seqs = {}
            for byte_seq, count in chunk_byte_seqs.items():
                merged_seq = self._merge_pair_in_sequence(list(byte_seq), best_pair, next_token_id)
                new_chunk_byte_seqs[tuple(merged_seq)] = count

            chunk_byte_seqs = new_chunk_byte_seqs
            next_token_id += 1

        # Step 4: Add special tokens
        if verbose:
            print("Step 4: Adding special tokens...")

        for special_token in SPECIAL_TOKENS:
            self.special_tokens[special_token] = next_token_id
            self.vocab[next_token_id] = special_token.encode('utf-8')
            next_token_id += 1

        self.vocab_size = next_token_id

        if verbose:
            print(f"Training complete! Final vocab size: {self.vocab_size}")
            print(f"  Base tokens: 256")
            print(f"  Merged tokens: {len(self.merges)}")
            print(f"  Special tokens: {len(self.special_tokens)}")

    def _merge_pair_in_sequence(self, seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        Merge all occurrences of a pair in a sequence.

        Args:
            seq: List of token IDs
            pair: Pair to merge (token_a, token_b)
            new_id: New token ID for the merged pair

        Returns:
            New sequence with pairs merged
        """
        if len(seq) < 2:
            return seq

        result = []
        i = 0
        while i < len(seq):
            # Check if we can merge at this position
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                result.append(new_id)
                i += 2  # Skip both tokens
            else:
                result.append(seq[i])
                i += 1

        return result

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add <|bos|> at the start

        Returns:
            List of token IDs
        """
        # Split text using regex pattern
        chunks = self.compiled_pattern.findall(text)

        all_ids = []

        if add_special_tokens and "<|bos|>" in self.special_tokens:
            all_ids.append(self.special_tokens["<|bos|>"])

        for chunk in chunks:
            # Convert chunk to bytes
            chunk_bytes = chunk.encode('utf-8')

            # Start with byte-level tokens
            ids = list(chunk_bytes)

            # Apply BPE merges iteratively
            while len(ids) >= 2:
                # Find the best pair to merge (lowest merge ID = earliest merge)
                best_pair = None
                best_idx = None
                best_merge_id = float('inf')

                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    if pair in self.merges:
                        merge_id = self.merges[pair]
                        if merge_id < best_merge_id:
                            best_pair = pair
                            best_idx = i
                            best_merge_id = merge_id

                # If no merge found, we're done
                if best_pair is None:
                    break

                # Apply the merge
                ids = ids[:best_idx] + [self.merges[best_pair]] + ids[best_idx + 2:]

            all_ids.extend(ids)

        return all_ids

    def encode_special(self, special_token: str) -> int:
        """
        Encode a special token.

        Args:
            special_token: Special token string (e.g., "<|bos|>")

        Returns:
            Token ID
        """
        if special_token not in self.special_tokens:
            raise ValueError(f"Unknown special token: {special_token}")
        return self.special_tokens[special_token]

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text
        """
        # Concatenate all token bytes
        result_bytes = b''
        for token_id in ids:
            if token_id in self.vocab:
                result_bytes += self.vocab[token_id]
            else:
                raise ValueError(f"Unknown token ID: {token_id}")

        # Decode to UTF-8 string
        return result_bytes.decode('utf-8', errors='replace')

    def save(self, save_dir: str) -> None:
        """
        Save the tokenizer to disk.

        Args:
            save_dir: Directory to save tokenizer files
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save as a single JSON file for simplicity
        tokenizer_data = {
            'merges': [(list(k), v) for k, v in self.merges.items()],  # Convert tuple keys to lists
            'vocab': {k: list(v) for k, v in self.vocab.items()},  # Convert bytes to lists
            'special_tokens': self.special_tokens,
            'pattern': self.pattern,
            'vocab_size': self.vocab_size,
        }

        save_path = os.path.join(save_dir, 'pure_python_tokenizer.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, indent=2)

        print(f"Saved tokenizer to {save_path}")

    @classmethod
    def load(cls, save_dir: str) -> 'PurePythonBPETokenizer':
        """
        Load a tokenizer from disk.

        Args:
            save_dir: Directory containing tokenizer files

        Returns:
            Loaded tokenizer instance
        """
        load_path = os.path.join(save_dir, 'pure_python_tokenizer.json')

        with open(load_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        tokenizer = cls()
        tokenizer.merges = {tuple(k): v for k, v in tokenizer_data['merges']}
        tokenizer.vocab = {int(k): bytes(v) for k, v in tokenizer_data['vocab'].items()}
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        tokenizer.pattern = tokenizer_data['pattern']
        tokenizer.compiled_pattern = regex.compile(tokenizer.pattern)
        tokenizer.vocab_size = tokenizer_data['vocab_size']

        print(f"Loaded tokenizer from {load_path}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Merges: {len(tokenizer.merges)}")
        print(f"  Special tokens: {len(tokenizer.special_tokens)}")

        return tokenizer

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size

    def get_special_tokens(self) -> List[str]:
        """Get list of special tokens."""
        return list(self.special_tokens.keys())

    def visualize_encoding(self, text: str) -> None:
        """
        Visualize how text is encoded (for debugging/learning).

        Args:
            text: Text to visualize
        """
        ids = self.encode(text)
        print(f"\nText: {repr(text)}")
        print(f"Token IDs: {ids}")
        print(f"Number of tokens: {len(ids)}")
        print("\nToken breakdown:")
        for i, token_id in enumerate(ids):
            token_bytes = self.vocab[token_id]
            token_str = token_bytes.decode('utf-8', errors='replace')
            print(f"  [{i}] ID={token_id:5d} | {repr(token_str):20s} | bytes={list(token_bytes)}")


# Convenience function to create and train a tokenizer
def train_tokenizer(text_iterator: Iterator[str],
                    vocab_size: int = 8192,
                    verbose: bool = True) -> PurePythonBPETokenizer:
    """
    Create and train a BPE tokenizer.

    Args:
        text_iterator: Iterator yielding text strings
        vocab_size: Target vocabulary size
        verbose: Whether to print progress

    Returns:
        Trained tokenizer
    """
    tokenizer = PurePythonBPETokenizer()
    tokenizer.train(text_iterator, vocab_size, verbose=verbose)
    return tokenizer


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("Pure Python BPE Tokenizer - Demo")
    print("=" * 70)

    # Create some sample training data
    training_texts = [
        "Hello, world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Python programming is fun and powerful.",
        "Natural language processing with BPE tokenization.",
        "GPT models use byte pair encoding.",
        "Transformers revolutionized NLP.",
        "Hello again! Testing repeated words: test test test.",
    ] * 100  # Repeat to have more training data

    # Train a small tokenizer
    print("\nTraining tokenizer on sample data...")
    tokenizer = train_tokenizer(iter(training_texts), vocab_size=512, verbose=True)

    # Test encoding and decoding
    test_text = "Hello, world! Testing BPE."
    print(f"\n{'-' * 70}")
    print("Testing encoding and decoding:")
    print(f"{'-' * 70}")

    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {repr(test_text)}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {repr(decoded)}")
    print(f"Match: {test_text == decoded}")

    # Visualize encoding
    tokenizer.visualize_encoding(test_text)

    # Test special tokens
    print(f"\n{'-' * 70}")
    print("Special tokens:")
    print(f"{'-' * 70}")
    for token in tokenizer.get_special_tokens():
        token_id = tokenizer.encode_special(token)
        print(f"  {token:20s} -> ID {token_id}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
