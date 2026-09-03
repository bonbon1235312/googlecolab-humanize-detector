"""Byte-level BPE utilities."""

import json
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_tokenizer(training_file: Path, artifact_dir: Path, vocab_size: int) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    corpus = artifact_dir / "training_corpus.txt"
    with training_file.open(encoding="utf-8") as source, corpus.open("w", encoding="utf-8") as target:
        for line in source:
            target.write(json.loads(line)["text"].replace("\n", " ") + "\n")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[str(corpus)], vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    tokenizer.save_model(str(artifact_dir))
    return artifact_dir


def load_tokenizer(artifact_dir: Path) -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer(str(artifact_dir / "vocab.json"), str(artifact_dir / "merges.txt"))
