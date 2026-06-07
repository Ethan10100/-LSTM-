"""
数据治理模块 — 数据质量检查、元数据管理、数据底板构建辅助

对应课程知识：
  - 数据采集、清洗、转换、整合与治理
  - 数据底板构建
  - 空间数据组织与管理
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd


# ── 数据质量检查 ──────────────────────────────────────────

def check_data_completeness(df: pd.DataFrame, required_columns: List[str]) -> Dict:
    """检查必填字段完整性"""
    result = {"status": "ok", "missing_columns": [], "null_counts": {}}
    for col in required_columns:
        if col not in df.columns:
            result["missing_columns"].append(col)
            result["status"] = "error"
        else:
            null_count = df[col].isna().sum()
            if null_count > 0:
                result["null_counts"][col] = int(null_count)
    return result


def check_value_range(df: pd.DataFrame, column: str,
                      min_val: float, max_val: float) -> Dict:
    """检查数值范围是否合理"""
    values = df[column].dropna()
    outliers = values[(values < min_val) | (values > max_val)]
    return {
        "column": column,
        "min_allowed": min_val,
        "max_allowed": max_val,
        "actual_min": float(values.min()),
        "actual_max": float(values.max()),
        "out_of_range_count": len(outliers),
        "outlier_indices": outliers.index.tolist()[:20],  # 最多返回20个
        "is_ok": len(outliers) == 0,
    }


def check_temporal_continuity(df: pd.DataFrame, time_column: str = "timestamp",
                              expected_freq: str = "1h") -> Dict:
    """检查时间序列连续性"""
    if time_column not in df.columns:
        return {"status": "error", "message": f"列 '{time_column}' 不存在"}

    times = pd.to_datetime(df[time_column].dropna()).sort_values()
    if len(times) < 2:
        return {"status": "ok", "message": "数据点不足，跳过连续性检查"}

    time_diffs = times.diff().dropna()
    expected = pd.Timedelta(expected_freq)
    gaps = time_diffs[time_diffs > expected * 1.5]

    total_points = len(times)
    gap_count = len(gaps)

    return {
        "status": "ok" if gap_count < total_points * 0.05 else "warning",
        "total_points": total_points,
        "time_start": str(times.iloc[0]),
        "time_end": str(times.iloc[-1]),
        "expected_frequency": expected_freq,
        "gap_count": gap_count,
        "max_gap": str(gaps.max()) if gap_count > 0 else "N/A",
        "gap_percentage": f"{gap_count / total_points * 100:.2f}%",
    }


def check_duplicate_records(df: pd.DataFrame, key_columns: List[str]) -> Dict:
    """检查重复记录"""
    if not all(c in df.columns for c in key_columns):
        return {"status": "error", "message": "指定的键列不存在"}
    dup_mask = df.duplicated(subset=key_columns, keep=False)
    dup_count = dup_mask.sum()
    return {
        "status": "ok" if dup_count == 0 else "warning",
        "duplicate_count": int(dup_count),
        "duplicate_indices": df[dup_mask].index.tolist()[:20],
    }


def run_full_quality_check(df: pd.DataFrame, value_column: str = "water_level",
                           time_column: str = "timestamp",
                           min_val: float = 5.0, max_val: float = 30.0,
                           station_column: Optional[str] = None) -> Dict:
    """运行完整的数据质量检查并返回报告"""
    report = {
        "check_time": datetime.now().isoformat(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "checks": {},
    }

    # 1. 完整性检查
    completeness = check_data_completeness(df, [time_column, value_column])
    report["checks"]["completeness"] = completeness

    # 2. 数值范围检查
    range_check = check_value_range(df, value_column, min_val, max_val)
    report["checks"]["value_range"] = range_check

    # 3. 时间连续性检查
    if time_column in df.columns:
        temporal = check_temporal_continuity(df, time_column)
        report["checks"]["temporal_continuity"] = temporal

    # 4. 重复检查
    dup_cols = [time_column]
    if station_column and station_column in df.columns:
        dup_cols.append(station_column)
    dup_check = check_duplicate_records(df, dup_cols)
    report["checks"]["duplicates"] = dup_check

    # 5. 基本统计
    if value_column in df.columns:
        vals = df[value_column].dropna()
        report["checks"]["statistics"] = {
            "count": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "q25": float(vals.quantile(0.25)),
            "median": float(vals.median()),
            "q75": float(vals.quantile(0.75)),
            "max": float(vals.max()),
        }

    # 综合判定
    all_ok = all(
        c.get("status", c.get("is_ok", "ok")) in ("ok", True)
        for c in report["checks"].values()
        if isinstance(c, dict)
    )
    report["overall_status"] = "pass" if all_ok else "issues_found"

    return report


# ── 元数据管理 ────────────────────────────────────────────

def generate_metadata(data_path: str, value_column: str = "water_level",
                      time_column: str = "timestamp",
                      description: str = "",
                      data_source: str = "",
                      coordinate_system: str = "",
                      ) -> Dict:
    """生成数据集元数据文档"""
    df = pd.read_csv(data_path)
    quality_report = run_full_quality_check(df, value_column, time_column)

    metadata = {
        "dataset_name": os.path.basename(data_path),
        "file_path": os.path.abspath(data_path),
        "description": description or "水位监测时间序列数据",
        "data_source": data_source or "模拟数据 / 课程实训",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "coordinate_system": coordinate_system or "WGS84 (EPSG:4326)",
        "record_count": len(df),
        "column_count": len(df.columns),
        "columns": [
            {"name": col, "dtype": str(df[col].dtype),
             "null_count": int(df[col].isna().sum()),
             "sample_values": df[col].dropna().head(5).tolist()}
            for col in df.columns
        ],
        "time_range": {
            "start": str(df[time_column].iloc[0]) if time_column in df.columns else "N/A",
            "end": str(df[time_column].iloc[-1]) if time_column in df.columns else "N/A",
        },
        "spatial_extent": None,
        "quality_report": quality_report,
    }

    # 如果数据包含经纬度，提取空间范围
    if "lon" in df.columns and "lat" in df.columns:
        metadata["spatial_extent"] = {
            "west": float(df["lon"].min()),
            "east": float(df["lon"].max()),
            "south": float(df["lat"].min()),
            "north": float(df["lat"].max()),
        }

    return metadata


def save_metadata(metadata: Dict, output_path: str) -> None:
    """保存元数据为JSON文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"元数据已保存: {output_path}")


# ── 数据底板统计 ──────────────────────────────────────────

def generate_data_catalog(data_dir: str = "data") -> pd.DataFrame:
    """生成数据底板目录（扫描data目录下所有CSV文件）"""
    if not os.path.isdir(data_dir):
        return pd.DataFrame()

    records = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(fpath)
            records.append({
                "文件名": fname,
                "路径": fpath,
                "记录数": len(df),
                "列数": len(df.columns),
                "列名": ", ".join(df.columns[:8]),
                "文件大小(KB)": round(os.path.getsize(fpath) / 1024, 1),
                "修改时间": datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M"),
            })
        except Exception as e:
            records.append({
                "文件名": fname,
                "路径": fpath,
                "记录数": -1,
                "列数": -1,
                "列名": f"读取失败: {e}",
                "文件大小(KB)": -1,
                "修改时间": "-",
            })

    return pd.DataFrame(records)


def compute_station_statistics(df: pd.DataFrame, value_column: str = "water_level",
                               station_column: str = "station_id") -> pd.DataFrame:
    """按监测站计算统计指标"""
    if station_column not in df.columns:
        return pd.DataFrame()

    stats = df.groupby(station_column).agg(
        记录数=(value_column, "count"),
        均值=(value_column, "mean"),
        标准差=(value_column, "std"),
        最小值=(value_column, "min"),
        最大值=(value_column, "max"),
        极差=(value_column, lambda x: x.max() - x.min()),
    ).reset_index()

    stats.columns = ["监测站ID", "记录数", "均值", "标准差", "最小值", "最大值", "极差"]
    for col in ["均值", "标准差", "最小值", "最大值", "极差"]:
        stats[col] = stats[col].round(4)

    return stats


if __name__ == "__main__":
    # 测试数据治理功能
    from data_utils import generate_sample_water_data
    generate_sample_water_data("data/water_level.csv", n_samples=500)

    df = pd.read_csv("data/water_level.csv")
    report = run_full_quality_check(df)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    metadata = generate_metadata("data/water_level.csv")
    save_metadata(metadata, "results/metadata.json")

    catalog = generate_data_catalog("data")
    print("\n数据底板目录:")
    print(catalog.to_string(index=False))
