"""
GIS空间数据工具模块 — 水位监测站地理信息、流域地图生成、空间分析辅助

研究区域：汉江中下游（丹江口—武汉段）
数据用途：为水位异常检测系统提供地理空间上下文
"""

import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── 汉江中下游水位监测站（模拟真实地理位置）──────────────────
# 坐标范围: 约 110.7°~114.4°E, 30.3°~32.8°N
# 沿汉江从丹江口至武汉，按流向排列

HANJIANG_STATIONS: List[Dict] = [
    {
        "id": "HJ-001", "name": "丹江口站", "river_section": "丹江口水库下游",
        "lon": 111.513, "lat": 32.548, "elevation_m": 145.0,
        "station_type": "水文站", "drainage_area_km2": 95200,
        "description": "丹江口水库坝下控制站，南水北调中线水源地核心监测点",
    },
    {
        "id": "HJ-002", "name": "老河口站", "river_section": "汉江中游",
        "lon": 111.675, "lat": 32.392, "elevation_m": 118.0,
        "station_type": "水位站", "drainage_area_km2": 105000,
        "description": "老河口市区段监测点，反映汉江出丹江口后水位变化",
    },
    {
        "id": "HJ-003", "name": "谷城站", "river_section": "汉江中游",
        "lon": 111.653, "lat": 32.267, "elevation_m": 108.0,
        "station_type": "水文站", "drainage_area_km2": 113000,
        "description": "南河入汉江汇合口上游监测点",
    },
    {
        "id": "HJ-004", "name": "襄阳站", "river_section": "汉江中游",
        "lon": 112.142, "lat": 32.025, "elevation_m": 93.0,
        "station_type": "水文站", "drainage_area_km2": 140000,
        "description": "襄阳市区汉江段，唐白河汇入前控制站",
    },
    {
        "id": "HJ-005", "name": "宜城站", "river_section": "汉江中游",
        "lon": 112.258, "lat": 31.720, "elevation_m": 78.0,
        "station_type": "水位站", "drainage_area_km2": 148000,
        "description": "蛮河汇入口附近监测点",
    },
    {
        "id": "HJ-006", "name": "钟祥站", "river_section": "汉江中下游",
        "lon": 112.588, "lat": 31.168, "elevation_m": 58.0,
        "station_type": "水文站", "drainage_area_km2": 155000,
        "description": "汉江中下游过渡段，皇庄水文站",
    },
    {
        "id": "HJ-007", "name": "沙洋站", "river_section": "汉江下游",
        "lon": 112.589, "lat": 30.710, "elevation_m": 42.0,
        "station_type": "水位站", "drainage_area_km2": 159000,
        "description": "汉江下游沙洋段，江汉平原北部监测点",
    },
    {
        "id": "HJ-008", "name": "潜江站", "river_section": "汉江下游",
        "lon": 112.900, "lat": 30.402, "elevation_m": 35.0,
        "station_type": "水文站", "drainage_area_km2": 163000,
        "description": "东荆河分流口附近，汉南平原入口监测点",
    },
    {
        "id": "HJ-009", "name": "仙桃站", "river_section": "汉江下游",
        "lon": 113.454, "lat": 30.328, "elevation_m": 28.0,
        "station_type": "水位站", "drainage_area_km2": 167000,
        "description": "仙桃城区段，杜家台分洪闸上游",
    },
    {
        "id": "HJ-010", "name": "汉川站", "river_section": "汉江下游",
        "lon": 113.839, "lat": 30.652, "elevation_m": 22.0,
        "station_type": "水文站", "drainage_area_km2": 172000,
        "description": "汉江入长江前最后控制站，距汉口约30km",
    },
    {
        "id": "HJ-011", "name": "武汉龙王庙站", "river_section": "汉江入江口",
        "lon": 114.283, "lat": 30.573, "elevation_m": 18.0,
        "station_type": "水位站", "drainage_area_km2": 174000,
        "description": "汉江与长江交汇处，武汉市防洪关键监测点",
    },
]


def get_stations_dataframe() -> pd.DataFrame:
    """将监测站信息转为DataFrame"""
    return pd.DataFrame(HANJIANG_STATIONS)


def get_station_by_id(station_id: str) -> Optional[Dict]:
    """根据ID获取单个监测站信息"""
    for s in HANJIANG_STATIONS:
        if s["id"] == station_id:
            return s
    return None


def get_station_bounds() -> Dict[str, float]:
    """获取所有监测站的经纬度范围"""
    lons = [s["lon"] for s in HANJIANG_STATIONS]
    lats = [s["lat"] for s in HANJIANG_STATIONS]
    return {
        "min_lon": min(lons) - 0.2, "max_lon": max(lons) + 0.2,
        "min_lat": min(lats) - 0.2, "max_lat": max(lats) + 0.2,
    }


def get_center() -> Tuple[float, float]:
    """获取研究区中心坐标"""
    bounds = get_station_bounds()
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2
    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    return center_lat, center_lon


def generate_georeferenced_water_data(
    output_path: str = "data/hanjiang_water_level.csv",
    n_days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成汉江流域多监测站的地理参考水位数据

    参数:
        output_path: 输出CSV路径
        n_days: 模拟天数
        seed: 随机种子

    返回:
        包含 timestamp, station_id, water_level, is_anomaly 等列的DataFrame
    """
    np.random.seed(seed)
    records = []

    start_date = pd.Timestamp("2024-06-01 00:00:00")
    timestamps = pd.date_range(start=start_date, periods=n_days * 24, freq="h")

    for station in HANJIANG_STATIONS:
        base_level = 20.0 - station["elevation_m"] * 0.02  # 高程越高基准水位越低
        station_lat = station["lat"]

        for i, ts in enumerate(timestamps):
            hour_of_day = ts.hour
            day_of_year = ts.dayofyear

            daily = 1.8 * np.sin(2 * np.pi * hour_of_day / 24)
            seasonal = 2.5 * np.sin(2 * np.pi * day_of_year / 365 + station_lat * 0.5)
            noise = np.random.normal(0, 0.25)
            water_level = base_level + daily + seasonal + noise

            records.append({
                "timestamp": ts,
                "station_id": station["id"],
                "station_name": station["name"],
                "lon": station["lon"],
                "lat": station["lat"],
                "elevation_m": station["elevation_m"],
                "water_level": round(water_level, 4),
                "is_anomaly": 0,
            })

    df = pd.DataFrame(records)

    # 注入约3%的异常值（跨站点随机）
    n_total = len(df)
    n_anomalies = int(n_total * 0.03)
    anomaly_indices = np.random.choice(n_total, n_anomalies, replace=False)

    for idx in anomaly_indices:
        row = df.iloc[idx]
        if np.random.random() > 0.5:
            df.at[idx, "water_level"] += np.random.uniform(5.0, 8.0)
        else:
            df.at[idx, "water_level"] -= np.random.uniform(5.0, 8.0)
        df.at[idx, "is_anomaly"] = 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"汉江流域地理参考水位数据已生成: {output_path}")
    print(f"  监测站: {len(HANJIANG_STATIONS)} 个")
    print(f"  时间跨度: {n_days} 天 ({len(timestamps)} 条/站)")
    print(f"  总记录: {n_total} 条, 异常点: {n_anomalies}")

    return df


def generate_single_station_data(
    station_id: str = "HJ-004",
    output_path: str = "data/water_level.csv",
    n_samples: int = 2000,
    seed: int = 42,
    anomaly_ratio: float = 0.05,
    anomaly_magnitude: float = 6.0,
) -> pd.DataFrame:
    """
    为单个监测站生成水位数据（兼容现有系统接口）

    参数:
        station_id: 监测站ID (HJ-001 ~ HJ-011)
        output_path: 输出路径
        n_samples: 样本数
        seed: 随机种子
        anomaly_ratio: 异常比例
        anomaly_magnitude: 异常幅度

    返回:
        包含 timestamp, water_level, is_anomaly 的DataFrame
    """
    station = get_station_by_id(station_id)
    if station is None:
        station = HANJIANG_STATIONS[3]  # 默认襄阳站

    np.random.seed(seed)
    timestamps = pd.date_range(start="2024-01-01 00:00:00", periods=n_samples, freq="h")

    time = np.arange(n_samples)
    base = 14.0 + (112.5 - station["lon"]) * 2.0  # 上游基准水位更高
    daily = 2.5 * np.sin(2 * np.pi * time / 24)
    weekly = 0.8 * np.sin(2 * np.pi * time / (24 * 7))
    noise = np.random.normal(0, 0.3, n_samples)

    water_level = base + daily + weekly + noise

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
        "is_anomaly": [1 if i in anomaly_indices else 0 for i in range(n_samples)],
    })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"监测站 {station['name']}({station_id}) 数据已生成: {output_path}")
    print(f"  坐标: ({station['lon']}, {station['lat']}), 高程: {station['elevation_m']}m")
    print(f"  样本: {n_samples}, 异常: {n_anomalies}")

    return df


def generate_research_area_geojson() -> Dict:
    """生成研究区域（汉江中下游）的GeoJSON边界"""
    stations = HANJIANG_STATIONS
    coords = [[s["lon"], s["lat"]] for s in stations]
    # 构建多边形缓冲区（简化版，实际应用中使用shapely）
    center_lon = np.mean([s["lon"] for s in stations])
    center_lat = np.mean([s["lat"] for s in stations])
    span_lon = max(s["lon"] - center_lon for s in stations) + 0.5
    span_lat = max(s["lat"] - center_lat for s in stations) + 0.5

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "汉江中下游研究区",
                    "description": "丹江口—武汉段，11个水位监测站",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [center_lon - span_lon, center_lat - span_lat],
                        [center_lon + span_lon, center_lat - span_lat],
                        [center_lon + span_lon, center_lat + span_lat],
                        [center_lon - span_lon, center_lat + span_lat],
                        [center_lon - span_lon, center_lat - span_lat],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "监测站"},
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": [[s["lon"], s["lat"]] for s in stations],
                },
            },
        ],
    }


if __name__ == "__main__":
    # 测试：生成汉江流域多站点数据
    df = generate_georeferenced_water_data()
    print("\n监测站信息:")
    print(get_stations_dataframe()[["id", "name", "lon", "lat", "elevation_m"]].to_string(index=False))
