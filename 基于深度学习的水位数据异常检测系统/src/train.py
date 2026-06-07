import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data_utils import prepare_data, save_scaler
from model import LSTMAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train_model(data_path: str, model_path: str, scaler_path: str, epochs: int,
                batch_size: int, seq_len: int, lr: float, val_split: float,
                patience: int, loss_plot_path: str):
    sequences, _, scaler = prepare_data(data_path, seq_len)
    logger.info("Loaded %d sequences from %s", len(sequences), data_path)

    # 划分训练集和验证集
    n_val = int(len(sequences) * val_split)
    n_train = len(sequences) - n_val
    train_seqs, val_seqs = sequences[:n_train], sequences[n_train:]
    logger.info("Train sequences: %d, Validation sequences: %d", n_train, n_val)

    train_dataset = TensorDataset(torch.from_numpy(train_seqs))
    val_dataset = TensorDataset(torch.from_numpy(val_seqs))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cfg = Config()
    device = cfg.device
    logger.info("Using device: %s", device)

    model = LSTMAutoencoder(
        input_size=1, hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= n_train
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                out = model(x)
                loss = criterion(out, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= n_val
        val_losses.append(val_loss)

        logger.info("Epoch %3d/%d | Train Loss: %.6f | Val Loss: %.6f",
                    epoch, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            save_scaler(scaler, scaler_path)
            logger.info("  -> Best model saved (val_loss=%.6f)", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    logger.info("Training finished. Best val loss: %.6f", best_val_loss)
    logger.info("Model saved to: %s", model_path)
    logger.info("Scaler saved to: %s", scaler_path)
    plot_training_loss(train_losses, val_losses, loss_plot_path)


def plot_training_loss(train_losses: list, val_losses: list, output_path: str):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="训练损失", color="blue")
    plt.plot(epochs, val_losses, label="验证损失", color="orange")
    plt.xlabel("训练轮数 (Epoch)")
    plt.ylabel("MSE 损失")
    plt.title("训练/验证损失变化曲线")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("训练损失曲线已保存到: %s", output_path)


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder for water level anomaly detection")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", default="models/lstm_autoencoder.pth")
    parser.add_argument("--scaler-path", default="models/scaler.pkl")
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--seq-len", type=int, default=cfg.seq_len)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--val-split", type=float, default=cfg.val_split)
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--loss-plot", default="results/training_loss.png")
    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        val_split=args.val_split,
        patience=args.patience,
        loss_plot_path=args.loss_plot,
    )


if __name__ == "__main__":
    main()
