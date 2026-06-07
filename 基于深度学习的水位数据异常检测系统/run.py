#!/usr/bin/env python
"""
水位数据异常检测系统 —— 一键运行完整流程

用法:
    python run.py                          # 使用默认参数运行完整流程
    python run.py --skip-generate          # 跳过数据生成（使用已有数据）
    python run.py --skip-train             # 跳过训练（使用已有模型）
    python run.py --threshold-factor 2.0   # 调整异常检测灵敏度
"""

import argparse
import logging
import sys

# 将 src 加入路径，确保模块可导入
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run")


def main():
    parser = argparse.ArgumentParser(description="水位数据异常检测系统")
    parser.add_argument("--skip-generate", action="store_true", help="跳过数据生成步骤")
    parser.add_argument("--skip-train", action="store_true", help="跳过模型训练步骤")
    parser.add_argument("--data-path", default="data/water_level.csv")
    parser.add_argument("--model-path", default="models/lstm_autoencoder.pth")
    parser.add_argument("--scaler-path", default="models/scaler.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-factor", type=float, default=1.5)
    parser.add_argument("--output-csv", default="results/anomaly_results.csv")
    parser.add_argument("--output-plot", default="results/anomaly_plot.png")
    args = parser.parse_args()

    # Step 1: 生成示例数据
    if not args.skip_generate:
        logger.info("=" * 50)
        logger.info("Step 1/3: 生成示例水位数据")
        logger.info("=" * 50)
        from data_utils import generate_sample_water_data
        generate_sample_water_data(output_path=args.data_path)
    else:
        logger.info("跳过数据生成，使用已有数据: %s", args.data_path)

    # Step 2: 训练模型
    if not args.skip_train:
        logger.info("=" * 50)
        logger.info("Step 2/3: 训练LSTM自编码器模型")
        logger.info("=" * 50)
        from train import train_model
        train_model(
            data_path=args.data_path,
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            val_split=0.2,
            patience=10,
            loss_plot_path="results/training_loss.png",
        )
    else:
        logger.info("跳过模型训练，使用已有模型: %s", args.model_path)

    # Step 3: 异常检测与评估
    logger.info("=" * 50)
    logger.info("Step 3/3: 执行异常检测与评估")
    logger.info("=" * 50)
    from detect import detect_anomalies, plot_results
    result, threshold = detect_anomalies(
        data_path=args.data_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        seq_len=args.seq_len,
        threshold_factor=args.threshold_factor,
    )
    result.to_csv(args.output_csv, index=False)
    logger.info("检测结果已保存到: %s", args.output_csv)
    logger.info("异常阈值: %.6f", threshold)
    plot_results(result, threshold, args.output_plot)


if __name__ == "__main__":
    main()
