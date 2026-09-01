import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class GenomicEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, pad_token_id: int = 0, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embed(input_ids)
        embeddings = self.pos_encoder(embeddings)
        return self.layer_norm(embeddings)

class VariantTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 69,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        max_len: int = 512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = GenomicEmbedding(vocab_size, embed_dim, pad_token_id=pad_token_id, max_len=max_len, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        hidden_states = self.embedding(input_ids)
        encoder_outputs = self.transformer_encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)
        
        cls_rep = encoder_outputs[:, 0, :]
        logits = self.classifier(cls_rep)
        probabilities = F.softmax(logits, dim=-1)
        lm_logits = self.lm_head(encoder_outputs)
        
        out = {
            "logits": logits,
            "probabilities": probabilities,
            "lm_logits": lm_logits,
            "cls_embedding": cls_rep
        }
        if return_embeddings:
            out["hidden_states"] = encoder_outputs
        return out
