import os
import pickle
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_water_level_data(csv_path: str, value_column: str = "water_level") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if value_column not in df.columns:
        raise ValueError(f"Value column '{value_column}' not found in CSV")
    df = df[[value_column]].copy()
    df = df.dropna().reset_index(drop=True)
    return df


def load_data_with_labels(csv_path: str, value_column: str = "water_level",
                          label_column: str = "is_anomaly") -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """加载水位数据，同时读取标签列（如果存在）"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if value_column not in df.columns:
        raise ValueError(f"Value column '{value_column}' not found in CSV")
    df = df.dropna(subset=[value_column]).reset_index(drop=True)
    labels = df[label_column].copy() if label_column in df.columns else None
    return df[[value_column]].copy(), labels


def normalize_series(values: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    values = values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def create_sequences(values: np.ndarray, seq_len: int) -> np.ndarray:
    sequences = []
    for i in range(len(values) - seq_len + 1):
        sequences.append(values[i: i + seq_len])
    return np.array(sequences, dtype=np.float32)


def prepare_data(csv_path: str, seq_len: int = 30,
                 value_column: str = "water_level",
                 clean_only: bool = True,
                 label_column: str = "is_anomaly") -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """准备训练/检测数据。

    Args:
        clean_only: 若为True，Scaler仅拟合正常样本，且只返回正常数据的序列。
                     这确保模型只学习正常模式，检测时对异常更灵敏。
    """
    df, labels = load_data_with_labels(csv_path, value_column, label_column)

    if clean_only and labels is not None:
        clean_mask = labels == 0
        clean_values = df.loc[clean_mask, value_column].values
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaler.fit(clean_values.reshape(-1, 1))
        all_scaled = scaler.transform(df[value_column].values.reshape(-1, 1))
        sequences = create_sequences(all_scaled, seq_len)

        # 标记包含异常点的序列：序列中任一点为异常则该序列被排除
        is_anomaly_arr = labels.values
        seq_has_anomaly = np.array([
            np.any(is_anomaly_arr[i:i + seq_len] == 1)
            for i in range(len(sequences))
        ])
        clean_seq_mask = ~seq_has_anomaly
        return sequences[clean_seq_mask], df[value_column].values[seq_len - 1:][clean_seq_mask], scaler
    else:
        scaled, scaler = normalize_series(df[value_column].values)
        sequences = create_sequences(scaled, seq_len)
        return sequences, df[value_column].values[seq_len - 1:], scaler


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> MinMaxScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_sample_water_data(output_path: str = "data/water_level.csv",
                               n_samples: int = 2000, seed: int = 42,
                               anomaly_ratio: float = 0.05,
                               anomaly_magnitude: float = 6.0) -> pd.DataFrame:
    """
    生成示例水位时间序列数据（包含异常值）
    """
    np.random.seed(seed)

    timestamps = pd.date_range(start="2024-01-01 00:00:00", periods=n_samples, freq="h")

    time = np.arange(n_samples)
    daily_pattern = 2.5 * np.sin(2 * np.pi * time / 24)
    weekly_pattern = 0.8 * np.sin(2 * np.pi * time / (24 * 7))
    trend = 0.001 * time
    noise = np.random.normal(0, 0.3, n_samples)

    water_level = 12.0 + daily_pattern + weekly_pattern + trend + noise

    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    for idx in anomaly_indices:
        if np.random.random() > 0.5:
            water_level[idx] += np.random.uniform(anomaly_magnitude, anomaly_magnitude + 2.0)
        else:
            water_level[idx] -= np.random.uniform(anomaly_magnitude, anomaly_magnitude + 2.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "water_level": water_level,
        "is_anomaly": [1 if i in anomaly_indices else 0 for i in range(n_samples)]
    })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"示例数据已生成并保存到: {output_path}")
    print(f"总样本数: {n_samples}, 异常点数: {n_anomalies}")

    return df


if __name__ == "__main__":
    generate_sample_water_data()
