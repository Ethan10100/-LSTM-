import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(decoder_input)
        out = self.output_layer(decoded)
        return out


def compute_reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """整段序列的平均重构误差（用于训练）"""
    return torch.mean((original - reconstructed) ** 2, dim=(1, 2))


def compute_pointwise_error(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """逐时间步的重构误差 — 取每个序列最后一个时间步的误差作为该点的异常分数"""
    return ((original - reconstructed) ** 2).squeeze(-1)[:, -1]
