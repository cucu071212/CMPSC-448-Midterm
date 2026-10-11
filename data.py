"""Turn the processed Arena CSVs into padded token-id tensors for the CNN / RNN.

Shared by both models so they see exactly the same inputs.

Usage (Colab, after mounting Drive):
    from src.data import build_loaders
    loaders, vocab = build_loaders(DATA_DIR, mode="output")
    for x, y in loaders["train"]:   # x: (B, max_len) LongTensor, y: (B,) LongTensor
        ...

mode (for RQ2):
    "output" -> LLM_output only
    "input"  -> LLM_Input only
    "both"   -> LLM_Input (first max_prompt tokens) + <sep> + LLM_output
"""
import json
import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PAD, UNK, SEP, NL = "<pad>", "<unk>", "<sep>", "<nl>"
SPECIALS = [PAD, UNK, SEP, NL]          # PAD must be index 0 (models use pad_idx=0)

# newline | word | run of the same symbol ("**", "##", "---", "—", "’", emoji, ...)
TOKEN_RE = re.compile(r"\n|\w+|([^\w\s])\1*")


def tokenize(text, max_chars=None):
    """Lowercase, keep punctuation / markdown / emoji as tokens, newline -> <nl>."""
    if max_chars:
        text = text[:max_chars]          # skip tokenizing the tail we will cut anyway
    return [NL if m.group(0) == "\n" else m.group(0)
            for m in TOKEN_RE.finditer(text.lower())]


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=2, max_size=30000):
        counts = Counter(t for toks in token_lists for t in toks)
        words = [t for t, c in counts.most_common() if c >= min_freq and t not in SPECIALS]
        return cls(SPECIALS + words[: max_size - len(SPECIALS)])

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


def load_split(data_dir, split):
    # keep_default_na=False: a response like "NA" stays text instead of NaN
    return pd.read_csv(f"{data_dir}/{split}.csv", keep_default_na=False)


def texts_to_tokens(df, mode, max_len, max_prompt=128):
    max_chars = max_len * 10             # generous upper bound on characters needed
    if mode == "output":
        return [tokenize(t, max_chars) for t in df["LLM_output"]]
    if mode == "input":
        return [tokenize(t, max_chars) for t in df["LLM_Input"]]
    if mode == "both":
        return [tokenize(p, max_prompt * 10)[:max_prompt] + [SEP] + tokenize(o, max_chars)
                for p, o in zip(df["LLM_Input"], df["LLM_output"])]
    raise ValueError(f"unknown mode: {mode}")


def encode_pad(token_lists, vocab, max_len):
    """Truncate / right-pad every sequence to max_len -> LongTensor (N, max_len)."""
    out = torch.zeros(len(token_lists), max_len, dtype=torch.long)   # 0 = <pad>
    for i, toks in enumerate(token_lists):
        ids = vocab.encode(toks[:max_len])
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def build_loaders(data_dir, mode="output", max_len=512, batch_size=64,
                  min_freq=2, max_size=30000, vocab=None, num_workers=2):
    """Return ({"train", "validation", "test"} -> DataLoader, vocab).

    The vocabulary is built from the training split only, unless one is passed in
    (e.g. Vocab.load(...) so the CNN and RNN share the same vocabulary file).
    """
    tokens, labels = {}, {}
    for split in ["train", "validation", "test"]:
        df = load_split(data_dir, split)
        tokens[split] = texts_to_tokens(df, mode, max_len)
        labels[split] = torch.tensor(df["label"].values, dtype=torch.long)

    if vocab is None:
        vocab = Vocab.build(tokens["train"], min_freq=min_freq, max_size=max_size)

    loaders = {}
    for split in tokens:
        ds = TensorDataset(encode_pad(tokens[split], vocab, max_len), labels[split])
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, pin_memory=True)
    return loaders, vocab


if __name__ == "__main__":
    sample = "Certainly! Here’s a **quick** list:\n\n1.  First — done ✅\n## Notes\n---"
    print(tokenize(sample))
