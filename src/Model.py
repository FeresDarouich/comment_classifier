# model.py
# Tiny Transformer Encoder for text classification (CPU-friendly)
# Expects:
#   input_ids:      [B, T] int64
#   attention_mask: [B, T] with 1 for tokens, 0 for padding
# Outputs:
#   logits: [B, num_classes]

from __future__ import annoatations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    max_len: int = 64
    num_classes: int = 2
    d_model: int= 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 2
    dropout: float = 0.1
    pad_id: int = 0 # must match tockenizer "<pad>" id
    use_cls_pooling: bool= True # else masked mean pooling

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int,n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads= n_heads,
            dropout= dropout,
            batch_first= True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out, _ = self.attn(x,x,x,key_padding_mask= key_padding_mask, need_weights = False)
        x = self.ln1(x+self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x+self.drop(ff_out))
        return x
    
class TinyTransformerClassifier(nn.Module):
    def __init__(self, cfg:ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_module must be divisible by n_heads")
        self.cfg= cfg
        self.token_emb = nn.Embedding(cfg.vocab_size,cfg.d_model, padding_idx=cfg.pad_id)
        # learned positionalembeddings
        self.pos_emb = nn.Embedding(cfg.max_len,cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(cfg.d_model,cfg.n_heads,cfg.d_ff,cfg.dropout) for _ in range(cfg.n_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model,cfg.num_classes),
        )


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B,T] (torch.long)
        attention_mask: [B, T] (0/1 or bool) 1 = token , 0 = pad
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        B,T = input_ids.shape
        if T > self.cfg.max_len:
            raise ValueError(f" sequence length {T} exceeds max_len {self.cfg.max_len}")
        # Build positions [0..T-1]
        pos = torch.arange(T,device=input_ids.device).unsqueeze(0).expand(B,T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        # MultiheadAttention expects key_paddin_mask with True on pad positions
        # attention_mask: 1 token,  pad -> pad mask true when pad
        pad_mask = (attention_mask == 0)

        for blk in self.blocks:
            x = blk(x,key_padding_mask = pad_mask)
        # pooling
        if self.cfg.use_cls_pooling:
             # asume <cls> is at position 0
             pooled = x[:,0,:] # [B,D]
        else:
            # masked mean pooling over tokens
            mask = attention_mask.unsqueeze(-1).float() # [B,T,1]
            summed = (x * mask).sum(dim=1)              # [B, D]
            denom = mask.sum(dim=1).clamp(min=1.0)      # [B,1]
            pooled = summed / denom

        logits = self.classifier(pooled)  # [B,C]
        return logits 
    
def build_model(
        vocab_size: int,
        max_len: int = 64,
        num_classes: int = 2,
        pad_id: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_cls_pooling: bool = True,
    ) -> TinyTransformerClassifier:
        cfg = ModelConfig(
            vocab_size=vocab_size,
            max_len=max_len,
            num_classes=num_classes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_id=pad_id,
            use_cls_pooling=use_cls_pooling,
        )
        return TinyTransformerClassifier(cfg)

if __name__ == "__main__":
    # Minimal sanity test
    cfg = ModelConfig(vocab_size=100, max_len=12, num_classes=2)
    model = TinyTransformerClassifier(cfg)

    input_ids = torch.tensor([[2, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    logits = model(input_ids, attention_mask)
    print("logits shape:", logits.shape)  # [1, 2]