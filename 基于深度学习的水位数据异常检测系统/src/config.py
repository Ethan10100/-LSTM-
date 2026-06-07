from dataclasses import dataclass
import torch


@dataclass
class Config:
    seq_len: int = 30
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    threshold_factor: float = 1.5
    val_split: float = 0.2
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
