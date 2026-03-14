"""Calibration data loading for knowledge distillation repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch


def load_calibration_data(
    path: str,
    tokenizer=None,
    max_samples: int = 100,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """Load calibration data from file.

    Supports:
    - .jsonl files with a "text" field per line
    - .txt files with one sample per line
    - .pt files containing pre-tokenized tensors (list of tensors or single tensor)

    Args:
        path: Path to calibration data file.
        tokenizer: HuggingFace-style tokenizer (must have __call__ returning
            input_ids). Required for .jsonl and .txt files.
        max_samples: Maximum number of samples to load.
        max_length: Maximum sequence length (tokens). Sequences are truncated.

    Returns:
        List of 1-D int64 tensors, each containing token IDs.

    Raises:
        ValueError: If text file provided without tokenizer, or unsupported format.
        FileNotFoundError: If path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration data not found: {path}")

    suffix = p.suffix.lower()

    if suffix == ".pt":
        return _load_pt(p, max_samples, max_length)
    elif suffix == ".jsonl":
        return _load_jsonl(p, tokenizer, max_samples, max_length)
    elif suffix == ".txt":
        return _load_txt(p, tokenizer, max_samples, max_length)
    else:
        raise ValueError(
            f"Unsupported calibration file format: {suffix}. "
            "Use .jsonl, .txt, or .pt"
        )


def _load_pt(
    path: Path, max_samples: int, max_length: int
) -> List[torch.Tensor]:
    """Load pre-tokenized .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(data, torch.Tensor):
        # Single tensor: treat rows as samples
        if data.dim() == 1:
            data = [data]
        elif data.dim() == 2:
            data = [row for row in data]
        else:
            raise ValueError(
                f"Expected 1-D or 2-D tensor in .pt file, got {data.dim()}-D"
            )
    elif isinstance(data, (list, tuple)):
        # Already a list of tensors — keep as-is
        pass
    else:
        raise ValueError(
            f"Expected tensor or list of tensors in .pt file, got {type(data)}"
        )

    result = []
    for t in data[:max_samples]:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.long)
        t = t.long().flatten()[:max_length]
        result.append(t)
    return result


def _load_jsonl(
    path: Path, tokenizer, max_samples: int, max_length: int
) -> List[torch.Tensor]:
    """Load .jsonl file with 'text' field per line."""
    if tokenizer is None:
        raise ValueError(
            "tokenizer is required when loading .jsonl calibration data"
        )
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                texts.append(text)
            if len(texts) >= max_samples:
                break
    return _tokenize_texts(texts, tokenizer, max_length)


def _load_txt(
    path: Path, tokenizer, max_samples: int, max_length: int
) -> List[torch.Tensor]:
    """Load .txt file with one sample per line."""
    if tokenizer is None:
        raise ValueError(
            "tokenizer is required when loading .txt calibration data"
        )
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
            if len(texts) >= max_samples:
                break
    return _tokenize_texts(texts, tokenizer, max_length)


def _tokenize_texts(
    texts: List[str], tokenizer, max_length: int
) -> List[torch.Tensor]:
    """Tokenize a list of text strings."""
    result = []
    for text in texts:
        encoded = tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt"
        )
        ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
        result.append(ids)
    return result


def generate_random_calibration(
    vocab_size: int = 32000,
    num_samples: int = 50,
    seq_length: int = 128,
) -> List[torch.Tensor]:
    """Generate random token sequences for calibration.

    This is a fallback when no real calibration data is available. The random
    tokens won't produce meaningful activations, but they allow the distillation
    loop to run and provide *some* gradient signal.

    Args:
        vocab_size: Size of the token vocabulary.
        num_samples: Number of random sequences to generate.
        seq_length: Length of each sequence.

    Returns:
        List of 1-D int64 tensors with random token IDs in [0, vocab_size).
    """
    return [
        torch.randint(0, vocab_size, (seq_length,), dtype=torch.long)
        for _ in range(num_samples)
    ]
