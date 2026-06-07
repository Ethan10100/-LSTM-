import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from config import Config
from data_utils import create_sequences, load_data_with_labels, load_scaler
from model import LSTMAutoencoder, compute_pointwise_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def detect_anomalies(data_path: str, model_path: str, scaler_path: str,
                     seq_len: int, threshold_factor: float,
                     value_column: str = "water_level") -> pd.DataFrame:
    # 加载数据和标签
    df_values, labels = load_data_with_labels(data_path, value_column)
    logger.info("Loaded %d data points from %s", len(df_values), data_path)

    # 加载训练时保存的scaler并进行归一化
    scaler = load_scaler(scaler_path)
    values = df_values[value_column].values.reshape(-1, 1)
    scaled = scaler.transform(values)
    sequences = create_sequences(scaled, seq_len)

    # 加载模型
    cfg = Config()
    device = cfg.device
    model = LSTMAutoencoder(
        input_size=1, hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = TensorDataset(torch.from_numpy(sequences))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            out = model(x)
            batch_errors = compute_pointwise_error(x, out)
            errors.extend(batch_errors.cpu().numpy().tolist())

    errors = np.array(errors)
    threshold = np.mean(errors) + threshold_factor * np.std(errors)
    anomaly_flags = errors > threshold

    offset = seq_len - 1
    result = pd.DataFrame({
        "index": np.arange(len(df_values))[offset:],
        value_column: df_values[value_column].values[offset:],
        "reconstruction_error": errors,
        "is_anomaly": anomaly_flags.astype(int),
    })

    # 评估指标
    if labels is not None:
        gt = labels.values[offset:]
        evaluate_detection(gt, anomaly_flags)

    return result, threshold


def evaluate_detection(ground_truth: np.ndarray, predicted: np.ndarray):
    """计算并输出异常检测评估指标"""
    tp = np.sum((ground_truth == 1) & (predicted == 1))
    tn = np.sum((ground_truth == 0) & (predicted == 0))
    fp = np.sum((ground_truth == 0) & (predicted == 1))
    fn = np.sum((ground_truth == 1) & (predicted == 0))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info("=" * 55)
    logger.info("  异常检测评估报告")
    logger.info("=" * 55)
    logger.info("  混淆矩阵:")
    logger.info("                 预测正常    预测异常")
    logger.info("  实际正常       %6d      %6d", tn, fp)
    logger.info("  实际异常       %6d      %6d", fn, tp)
    logger.info("-" * 55)
    logger.info("  准确率 (Accuracy):  %.4f", accuracy)
    logger.info("  精确率 (Precision): %.4f", precision)
    logger.info("  召回率 (Recall):    %.4f", recall)
    logger.info("  F1分数 (F1-Score):  %.4f", f1)
    logger.info("=" * 55)
    logger.info("  总样本数: %d | 检测异常数: %d | 真实异常数: %d",
                total, int(np.sum(predicted)), int(np.sum(ground_truth)))


def plot_results(df: pd.DataFrame, threshold: float, output_path: str):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df["index"], df["water_level"], label="水位", color="blue", linewidth=1.5)
    anomaly_df = df[df["is_anomaly"] == 1]
    ax1.scatter(anomaly_df["index"], anomaly_df["water_level"],
                color="red", label="异常点", marker="o", s=60, edgecolor="black")
    ax1.set_title("水位数据异常检测结果")
    ax1.set_ylabel("水位值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["index"], df["reconstruction_error"],
             label="重构误差", color="orange", linewidth=1.5)
    ax2.axhline(y=threshold, color="red", linestyle="--",
                label=f"异常阈值 ({threshold:.6f})")
    ax2.scatter(anomaly_df["index"], anomaly_df["reconstruction_error"],
                color="red", marker="x", s=50, label="异常点误差")
    ax2.set_xlabel("时间索引")
    ax2.set_ylabel("重构误差")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("检测结果图已保存到: %s", output_path)


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Detect anomalies in water level data")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", default="models/lstm_autoencoder.pth")
    parser.add_argument("--scaler-path", default="models/scaler.pkl")
    parser.add_argument("--seq-len", type=int, default=cfg.seq_len)
    parser.add_argument("--threshold-factor", type=float, default=cfg.threshold_factor)
    parser.add_argument("--output-csv", default="results/anomaly_results.csv")
    parser.add_argument("--output-plot", default="results/anomaly_plot.png")
    args = parser.parse_args()

    result, threshold = detect_anomalies(
        args.data_path, args.model_path, args.scaler_path,
        args.seq_len, args.threshold_factor,
    )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    logger.info("检测结果已保存到: %s", args.output_csv)
    logger.info("异常阈值: %.6f", threshold)
    plot_results(result, threshold, args.output_plot)


if __name__ == "__main__":
    main()
