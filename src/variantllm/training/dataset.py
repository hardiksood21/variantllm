import pandas as pd
import torch
from torch.utils.data import Dataset
from ..tokenization.genomic_tokenizer import GenomicTokenizer

class ClinVarDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: GenomicTokenizer, sequence_col: str = "mut_seq", label_col: str = "label"):
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.sequence_col = sequence_col
        self.label_col = label_col
        
        self.sequences = self.df[sequence_col].astype(str).tolist()
        self.labels = self.df[label_col].astype(int).tolist()
        self.encodings = self.tokenizer.batch_encode_plus(self.sequences, return_tensors="pt")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
