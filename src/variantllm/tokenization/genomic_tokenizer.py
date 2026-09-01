import re
from typing import List, Dict, Union
import torch

class GenomicTokenizer:
    def __init__(self, kmer_size: int = 3, stride: int = 1, max_length: int = 256):
        self.kmer_size = kmer_size
        self.stride = stride
        self.max_length = max_length
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<MASK>": 4}
        self.vocab = dict(self.special_tokens)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._build_vocab()

    def _build_vocab(self):
        bases = ["A", "C", "G", "T"]
        if self.kmer_size == 3:
            kmers = [b1 + b2 + b3 for b1 in bases for b2 in bases for b3 in bases]
        else:
            kmers = bases
        for kmer in kmers:
            if kmer not in self.vocab:
                idx = len(self.vocab)
                self.vocab[kmer] = idx
                self.inv_vocab[idx] = kmer

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<PAD>"]

    @property
    def cls_token_id(self) -> int:
        return self.special_tokens["<CLS>"]

    @property
    def sep_token_id(self) -> int:
        return self.special_tokens["<SEP>"]

    def tokenize_sequence(self, sequence: str) -> List[str]:
        seq = re.sub(r"[^ACGTN]", "", sequence.upper())
        kmers = []
        for i in range(0, len(seq) - self.kmer_size + 1, self.stride):
            kmers.append(seq[i : i + self.kmer_size])
        return kmers

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize_sequence(sequence)
        token_ids = [self.vocab.get(t, self.special_tokens["<UNK>"]) for t in tokens]
        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length - 1] + [self.sep_token_id]
        return token_ids

    def batch_encode_plus(self, sequences: List[str], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        batch_ids = [self.encode(s, add_special_tokens=True) for s in sequences]
        max_len = min(max(len(ids) for ids in batch_ids), self.max_length)
        padded_ids = []
        attention_masks = []
        for ids in batch_ids:
            if len(ids) < max_len:
                pad_len = max_len - len(ids)
                p_ids = ids + [self.pad_token_id] * pad_len
                attn = [1] * len(ids) + [0] * pad_len
            else:
                p_ids = ids[:max_len]
                attn = [1] * max_len
            padded_ids.append(p_ids)
            attention_masks.append(attn)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
            }
        return {"input_ids": padded_ids, "attention_mask": attention_masks}
