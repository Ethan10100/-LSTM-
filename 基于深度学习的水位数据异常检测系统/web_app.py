"""
水位数据异常检测系统 — Web交互界面
启动方式: streamlit run web_app.py
"""

import os
import sys
import io
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# 确保 src 模块可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config
from data_utils import (
    create_sequences, load_scaler, load_data_with_labels,
    generate_sample_water_data, save_scaler,
)
from model import LSTMAutoencoder, compute_pointwise_error
from gis_utils import (
    HANJIANG_STATIONS, get_stations_dataframe, get_station_bounds, get_center,
    generate_georeferenced_water_data, generate_single_station_data,
    generate_research_area_geojson,
)
from data_governance import (
    run_full_quality_check, generate_metadata, save_metadata,
    generate_data_catalog, compute_station_statistics,
)

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="水位数据异常检测系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 缓存：加载模型 ────────────────────────────────────────
@st.cache_resource
def load_detection_model(model_path: str, scaler_path: str):
    cfg = Config()
    device = cfg.device
    model = LSTMAutoencoder(
        input_size=1, hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = load_scaler(scaler_path)
    return model, scaler, device


def run_detection(values: np.ndarray, model, scaler, device,
                  seq_len: int, threshold_factor: float, batch_size: int = 64):
    """对原始水位数据运行异常检测，返回结果DataFrame和阈值"""
    scaled = scaler.transform(values.reshape(-1, 1))
    sequences = create_sequences(scaled, seq_len)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(sequences))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
    df = pd.DataFrame({
        "序号": np.arange(len(values))[offset:],
        "水位值": values[offset:],
        "重构误差": np.round(errors, 8),
        "是否异常": ["⚠ 异常" if f else "✓ 正常" for f in anomaly_flags],
    })
    return df, threshold


# ── 侧边栏 ────────────────────────────────────────────────
st.sidebar.title("🌊 水位异常检测系统")
st.sidebar.markdown("---")

# 模型路径
st.sidebar.header("📦 模型配置")
model_path = st.sidebar.text_input("模型权重路径", "models/lstm_autoencoder.pth")
scaler_path = st.sidebar.text_input("Scaler路径", "models/scaler.pkl")

# 检测参数
st.sidebar.header("🎚️ 检测参数")
threshold_factor = st.sidebar.slider(
    "异常阈值因子", min_value=0.5, max_value=5.0, value=2.0, step=0.1,
    help="阈值 = mean + factor × std，值越大异常判定越严格（检出越少）"
)
seq_len = st.sidebar.number_input("序列长度", min_value=5, max_value=200, value=30, step=5)

# 模型就绪检查
model_ready = os.path.exists(model_path) and os.path.exists(scaler_path)
if not model_ready:
    st.sidebar.warning("⚠ 模型文件不存在，请先训练模型或调整路径")
else:
    st.sidebar.success("✅ 模型已就绪")

st.sidebar.markdown("---")
st.sidebar.caption(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── 主界面标题 ────────────────────────────────────────────
st.title("🌊 基于深度学习的水位数据异常检测系统")
st.markdown("使用 LSTM 自编码器检测水位时间序列中的异常值")

# ── 选项卡 ────────────────────────────────────────────────
tab_input, tab_results, tab_gis, tab_governance, tab_train = st.tabs(
    ["📥 数据输入", "📊 检测结果", "🗺️ GIS地图", "📋 数据治理", "🏋️ 模型训练"]
)

# ═══════════════════════════════════════════════════════════
# Tab 1: 数据输入
# ═══════════════════════════════════════════════════════════
with tab_input:
    st.header("输入水位数据")

    input_method = st.radio(
        "选择数据输入方式:",
        ["📁 上传CSV文件", "✏️ 手动输入数值", "🎲 使用示例数据"],
        horizontal=True,
    )

    input_values = None
    input_df = None

    if input_method == "📁 上传CSV文件":
        uploaded_file = st.file_uploader("上传包含水位数据的CSV文件", type=["csv"])
        if uploaded_file is not None:
            try:
                full_df = pd.read_csv(uploaded_file)
                st.success(f"已加载 {len(full_df)} 行数据")

                # 列选择
                columns = full_df.columns.tolist()
                value_col = st.selectbox("选择水位数值所在的列:", columns)
                input_values = full_df[value_col].dropna().values.astype(np.float64)
                input_df = full_df

                st.subheader("数据预览")
                st.dataframe(full_df.head(20), use_container_width=True)
                st.caption(f"共 {len(full_df)} 条记录 | 所选列: {value_col}")

                # 简单统计
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("数据量", len(input_values))
                col2.metric("均值", f"{np.mean(input_values):.3f}")
                col3.metric("标准差", f"{np.std(input_values):.3f}")
                col4.metric("极差", f"{np.max(input_values) - np.min(input_values):.3f}")
            except Exception as e:
                st.error(f"读取文件失败: {e}")

    elif input_method == "✏️ 手动输入数值":
        st.markdown("输入逗号或空格分隔的水位数值（至少需要 `seq_len+1` 个数据点）:")
        text_input = st.text_area(
            "水位数值:",
            placeholder="例如: 12.0, 12.3, 11.8, 12.1, 12.5, ...",
            height=150,
        )
        if text_input.strip():
            try:
                cleaned = text_input.replace("\n", ",").replace(" ", ",")
                input_values = np.array([float(x.strip()) for x in cleaned.split(",") if x.strip()])
                st.success(f"已解析 {len(input_values)} 个数据点")
                st.subheader("输入数据曲线")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=input_values, mode="lines+markers",
                    name="水位值", line=dict(color="#1f77b4", width=1.5),
                ))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                st.error(f"解析数值失败: {e}")

    else:  # 使用示例数据
        st.markdown("使用内置生成器创建模拟水位数据:")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_samples = st.number_input("样本数量", 500, 5000, 2000, 100)
        with col2:
            anomaly_pct = st.slider("异常比例(%)", 1, 20, 5)
        with col3:
            anomaly_mag = st.slider("异常幅度", 3.0, 12.0, 6.0, 0.5)

        if st.button("🎲 生成示例数据", use_container_width=True):
            with st.spinner("正在生成..."):
                df = generate_sample_water_data(
                    output_path="data/water_level.csv",
                    n_samples=n_samples,
                    anomaly_ratio=anomaly_pct / 100,
                    anomaly_magnitude=anomaly_mag,
                )
                input_values = df["water_level"].values
                input_df = df
            st.success(f"已生成 {n_samples} 条数据（含 {int(n_samples * anomaly_pct / 100)} 个异常点）")

            # 数据预览图
            fig = go.Figure()
            anomaly_mask = df["is_anomaly"] == 1
            fig.add_trace(go.Scatter(
                x=df.index, y=df["water_level"],
                mode="lines", name="水位", line=dict(color="#1f77b4", width=1),
            ))
            fig.add_trace(go.Scatter(
                x=df.index[anomaly_mask], y=df["water_level"][anomaly_mask],
                mode="markers", name="注入异常（参考）",
                marker=dict(color="red", size=8, symbol="x"),
            ))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # 开始检测按钮
    st.markdown("---")
    detect_btn = st.button(
        "🔍 开始异常检测", type="primary", use_container_width=True,
        disabled=(input_values is None or not model_ready),
    )

    if input_values is not None and not model_ready:
        st.warning("请先在侧边栏配置正确的模型路径，或切换到「模型训练」选项卡训练新模型")

# ═══════════════════════════════════════════════════════════
# Tab 2: 检测结果
# ═══════════════════════════════════════════════════════════
with tab_results:
    if "detection_done" not in st.session_state:
        st.session_state.detection_done = False

    # 触发检测
    if detect_btn and input_values is not None and model_ready:
        with st.spinner("正在加载模型并执行检测..."):
            model, scaler, device = load_detection_model(model_path, scaler_path)
            result_df, threshold = run_detection(
                input_values, model, scaler, device, seq_len, threshold_factor
            )
            st.session_state.result_df = result_df
            st.session_state.threshold = threshold
            st.session_state.detection_done = True
            st.session_state.input_values = input_values

    if st.session_state.detection_done:
        result_df = st.session_state.result_df
        threshold = st.session_state.threshold
        input_vals = st.session_state.input_values

        st.header("📊 检测结果")

        # ── 指标卡片 ──
        n_total = len(result_df)
        n_anomalies = (result_df["是否异常"] == "⚠ 异常").sum()
        anomaly_rate = n_anomalies / n_total * 100 if n_total > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("检测数据量", n_total)
        col2.metric("检出异常数", n_anomalies)
        col3.metric("异常比例", f"{anomaly_rate:.2f}%")
        col4.metric("异常阈值", f"{threshold:.6f}")
        col5.metric("平均重构误差", f"{result_df['重构误差'].mean():.6f}")

        # 如果数据中有真实标签，显示评估指标
        if input_df is not None and "is_anomaly" in input_df.columns:
            offset = seq_len - 1
            gt = input_df["is_anomaly"].values[offset:]
            pred = (result_df["是否异常"] == "⚠ 异常").values
            tp = int(np.sum((gt == 1) & (pred == 1)))
            tn = int(np.sum((gt == 0) & (pred == 0)))
            fp = int(np.sum((gt == 0) & (pred == 1)))
            fn = int(np.sum((gt == 1) & (pred == 0)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            st.markdown("---")
            st.subheader("🎯 评估指标（与真实标签对比）")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("精确率 Precision", f"{precision:.4f}")
            metric_col2.metric("召回率 Recall", f"{recall:.4f}")
            metric_col3.metric("F1分数", f"{f1:.4f}")
            metric_col4.metric("准确率 Accuracy", f"{(tp + tn) / n_total:.4f}")

            with st.expander("查看混淆矩阵"):
                cm_df = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    columns=["预测正常", "预测异常"],
                    index=["实际正常", "实际异常"],
                )
                st.table(cm_df)

        st.markdown("---")

        # ── 可视化图表 ──
        st.subheader("📈 水位数据与异常点")
        fig1 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("水位时间序列", "重构误差"),
            vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
        )

        is_anom = result_df["是否异常"] == "⚠ 异常"
        indices = result_df["序号"].values
        water = result_df["水位值"].values
        errors = result_df["重构误差"].values

        # 上: 水位曲线
        fig1.add_trace(go.Scatter(
            x=indices, y=water, mode="lines",
            name="水位值", line=dict(color="#1f77b4", width=1.5),
        ), row=1, col=1)
        if n_anomalies > 0:
            fig1.add_trace(go.Scatter(
                x=indices[is_anom], y=water[is_anom],
                mode="markers", name=f"异常点 ({n_anomalies}个)",
                marker=dict(color="#e74c3c", size=10, symbol="x", line=dict(width=2, color="darkred")),
            ), row=1, col=1)

        # 下: 重构误差
        normal_color = np.where(is_anom, "#e74c3c", "#2ecc71")
        fig1.add_trace(go.Bar(
            x=indices, y=errors,
            name="重构误差",
            marker_color=normal_color,
            opacity=0.7,
        ), row=2, col=1)
        fig1.add_hline(
            y=threshold, line_dash="dash", line_color="#e74c3c",
            annotation_text=f"阈值: {threshold:.6f}",
            annotation_position="top right",
            row=2, col=1,
        )

        fig1.update_xaxes(title_text="数据索引", row=2, col=1)
        fig1.update_yaxes(title_text="水位值", row=1, col=1)
        fig1.update_yaxes(title_text="重构误差", row=2, col=1)
        fig1.update_layout(
            height=600, hovermode="x unified",
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── 误差分布 ──
        st.markdown("---")
        col_dist, col_table = st.columns([1, 1])
        with col_dist:
            st.subheader("📊 重构误差分布")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=errors, nbinsx=50, marker_color="#3498db", opacity=0.75,
                name="误差分布",
            ))
            fig_hist.add_vline(
                x=threshold, line_dash="dash", line_color="#e74c3c", line_width=2,
                annotation_text="阈值",
            )
            fig_hist.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_table:
            st.subheader("📋 检测结果明细")
            st.dataframe(
                result_df.style.apply(
                    lambda row: ["background-color: #ffe0e0" if row["是否异常"] == "⚠ 异常" else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True,
                height=400,
            )

        # ── 下载按钮 ──
        st.markdown("---")
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        st.download_button(
            "📥 下载检测结果 (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"anomaly_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    elif not st.session_state.detection_done:
        st.info("👈 请先在「数据输入」选项卡中加载数据并点击「开始异常检测」")

# ═══════════════════════════════════════════════════════════
# Tab 3: 模型训练
# ═══════════════════════════════════════════════════════════
with tab_train:
    st.header("🏋️ 训练新模型")

    st.markdown("训练一个新的LSTM自编码器模型。训练数据应包含**正常水位**样本（异常标签=0的数据用于训练）。")

    train_data_path = st.text_input("训练数据路径", "data/water_level.csv")
    train_model_path = st.text_input("模型保存路径", "models/lstm_autoencoder.pth")
    train_scaler_path = st.text_input("Scaler保存路径", "models/scaler.pkl")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_epochs = st.number_input("训练轮数", 10, 200, 50, 10)
        train_lr = st.number_input("学习率", 1e-5, 1e-1, 1e-3, format="%.5f")
    with col2:
        train_batch = st.number_input("批次大小", 16, 256, 64, 16)
        train_val_split = st.slider("验证集比例", 0.05, 0.4, 0.2, 0.05)
    with col3:
        train_seq_len = st.number_input("序列长度", 5, 200, 30, 5)
        train_patience = st.number_input("早停耐心值", 5, 30, 10, 1)

    if st.button("🚀 开始训练", type="primary", use_container_width=True,
                 disabled=not os.path.exists(train_data_path)):
        with st.spinner("正在训练..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 重定向stdout捕获训练日志
            import logging
            logger = logging.getLogger("train")
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)

            status_text.text("正在加载数据并训练模型...")
            try:
                from train import train_model
                train_model(
                    data_path=train_data_path,
                    model_path=train_model_path,
                    scaler_path=train_scaler_path,
                    epochs=train_epochs,
                    batch_size=train_batch,
                    seq_len=train_seq_len,
                    lr=train_lr,
                    val_split=train_val_split,
                    patience=train_patience,
                    loss_plot_path="results/training_loss.png",
                )
                progress_bar.progress(100)
                status_text.text("")
                st.success(f"✅ 训练完成！模型已保存到 `{train_model_path}`")

                # 显示训练损失图
                if os.path.exists("results/training_loss.png"):
                    st.image("results/training_loss.png", caption="训练/验证损失曲线")
            except Exception as e:
                st.error(f"训练失败: {e}")
            finally:
                logger.removeHandler(handler)
    elif not os.path.exists(train_data_path):
        st.warning(f"训练数据文件不存在: `{train_data_path}`，请先生成示例数据或提供有效的CSV文件")

# ═══════════════════════════════════════════════════════════
# Tab 3: GIS地图
# ═══════════════════════════════════════════════════════════
with tab_gis:
    st.header("🗺️ 汉江中下游流域 — 水位监测站分布")
    st.markdown("研究区域：丹江口水库—武汉龙王庙，全长约650km，11个水位监测站")

    stations_df = get_stations_dataframe()
    center_lat, center_lon = get_center()

    # 使用Plotly Scattermapbox绘制地图
    fig_map = go.Figure()

    # 监测站点位
    fig_map.add_trace(go.Scattermapbox(
        lat=stations_df["lat"].tolist(),
        lon=stations_df["lon"].tolist(),
        mode="markers+text",
        marker=dict(size=14, color="#e74c3c", opacity=0.9),
        text=stations_df["name"].tolist(),
        textposition="top center",
        textfont=dict(size=12, color="#2c3e50"),
        name="水位监测站",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "站点ID: %{customdata[0]}<br>"
            "高程: %{customdata[1]}m<br>"
            "类型: %{customdata[2]}<br>"
            "<extra></extra>"
        ),
        customdata=stations_df[["id", "elevation_m", "station_type"]].values,
    ))

    # 汉江简化线（沿监测站连线）
    fig_map.add_trace(go.Scattermapbox(
        lat=stations_df["lat"].tolist(),
        lon=stations_df["lon"].tolist(),
        mode="lines",
        line=dict(width=3, color="#3498db"),
        name="汉江河道（示意）",
        hoverinfo="none",
    ))

    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=7,
        ),
        height=550,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # 监测站信息表
    st.markdown("---")
    st.subheader("📋 监测站详细信息")
    display_cols = ["id", "name", "river_section", "lon", "lat", "elevation_m",
                    "station_type", "drainage_area_km2", "description"]
    st.dataframe(
        stations_df[display_cols].rename(columns={
            "id": "站点ID", "name": "站名", "river_section": "河段",
            "lon": "经度", "lat": "纬度", "elevation_m": "高程(m)",
            "station_type": "站点类型", "drainage_area_km2": "集水面积(km²)",
            "description": "说明",
        }),
        use_container_width=True, hide_index=True,
    )

    # 生成汉江流域多站点数据
    st.markdown("---")
    st.subheader("📊 生成汉江流域多站点数据")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_days = st.number_input("模拟天数", 7, 365, 90, 7, key="gis_days")
    with col2:
        gis_output = st.text_input("输出路径", "data/hanjiang_water_level.csv", key="gis_output")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗺️ 生成汉江流域数据", use_container_width=True):
            with st.spinner("正在生成多站点地理参考数据..."):
                df = generate_georeferenced_water_data(gis_output, n_days=n_days)
            st.success(f"已生成 {len(df)} 条记录（{len(df['station_id'].unique())} 个站点 × {n_days} 天）")

    # 单个站点数据生成
    st.markdown("---")
    st.subheader("📍 生成单个监测站数据（用于模型训练）")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_station = st.selectbox(
            "选择监测站",
            [f"{s['id']} - {s['name']}" for s in HANJIANG_STATIONS],
            index=3,  # 默认襄阳站
            key="gis_station_select",
        )
        station_id_selected = selected_station.split(" - ")[0]
    with col_s2:
        single_output = st.text_input("输出路径", "data/water_level.csv", key="single_output")
    if st.button("📍 生成单站训练数据", use_container_width=True):
        with st.spinner(f"正在为 {station_id_selected} 生成数据..."):
            df = generate_single_station_data(
                station_id=station_id_selected,
                output_path=single_output,
                n_samples=2000,
            )
        st.success(f"已生成数据 → `{single_output}`")
        # 显示数据预览
        fig_preview = go.Figure()
        fig_preview.add_trace(go.Scatter(
            x=df.index[:200], y=df["water_level"][:200],
            mode="lines", name="水位", line=dict(color="#1f77b4", width=1),
        ))
        fig_preview.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0),
                                  title="前200条数据预览")
        st.plotly_chart(fig_preview, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# Tab 4: 数据治理
# ═══════════════════════════════════════════════════════════
with tab_governance:
    st.header("📋 数据治理与质量检查")

    gov_data_path = st.text_input("数据文件路径", "data/water_level.csv", key="gov_data_path")

    if os.path.exists(gov_data_path):
        try:
            gov_df = pd.read_csv(gov_data_path)
            st.success(f"已加载: {gov_data_path} ({len(gov_df)} 行 × {len(gov_df.columns)} 列)")

            # 数据质量检查
            if st.button("🔍 运行数据质量检查", use_container_width=True):
                with st.spinner("正在运行质量检查..."):
                    report = run_full_quality_check(gov_df)

                st.subheader("📊 质量检查报告")

                # 综合判定
                overall = report.get("overall_status", "unknown")
                if overall == "pass":
                    st.success("✅ 数据质量综合判定：通过")
                else:
                    st.warning("⚠️ 数据质量综合判定：发现问题")

                checks = report.get("checks", {})

                # 统计数据
                if "statistics" in checks:
                    stats = checks["statistics"]
                    st.markdown("**基本统计量**")
                    cols = st.columns(8)
                    metric_items = [
                        ("数据量", stats.get("count", 0)),
                        ("均值", f"{stats.get('mean', 0):.3f}"),
                        ("标准差", f"{stats.get('std', 0):.3f}"),
                        ("最小值", f"{stats.get('min', 0):.3f}"),
                        ("Q25", f"{stats.get('q25', 0):.3f}"),
                        ("中位数", f"{stats.get('median', 0):.3f}"),
                        ("Q75", f"{stats.get('q75', 0):.3f}"),
                        ("最大值", f"{stats.get('max', 0):.3f}"),
                    ]
                    for i, (label, val) in enumerate(metric_items):
                        cols[i].metric(label, val)

                # 完整性
                comp = checks.get("completeness", {})
                if comp.get("status") == "error":
                    st.error(f"❌ 缺少必填列: {comp.get('missing_columns', [])}")
                else:
                    nulls = comp.get("null_counts", {})
                    if nulls:
                        st.warning(f"⚠️ 存在空值: {nulls}")
                    else:
                        st.success("✅ 必填字段完整，无空值")

                # 数值范围
                vr = checks.get("value_range", {})
                if vr.get("is_ok"):
                    st.success(f"✅ 数值范围正常 [{vr.get('min_allowed')}, {vr.get('max_allowed')}]")
                else:
                    st.warning(f"⚠️ {vr.get('out_of_range_count')} 个值超出范围")

                # 时间连续性
                tc = checks.get("temporal_continuity", {})
                if tc.get("status") == "ok":
                    st.success(f"✅ 时间序列连续 ({tc.get('time_start')} ~ {tc.get('time_end')})")

                # 重复检查
                dc = checks.get("duplicates", {})
                if dc.get("status") == "ok":
                    st.success("✅ 无重复记录")

                # JSON报告
                with st.expander("查看完整JSON报告"):
                    st.json(report)

            # 数据底板目录
            st.markdown("---")
            st.subheader("📂 数据底板目录")
            catalog = generate_data_catalog("data")
            if not catalog.empty:
                st.dataframe(catalog, use_container_width=True, hide_index=True)
            else:
                st.info("data/ 目录下暂无CSV文件")

            # 元数据生成
            st.markdown("---")
            st.subheader("📝 生成元数据文档")
            if st.button("📝 生成元数据JSON", use_container_width=True):
                metadata = generate_metadata(gov_data_path)
                save_metadata(metadata, "results/metadata.json")
                st.success("元数据已保存到 results/metadata.json")
                with st.expander("查看元数据"):
                    st.json(metadata)

            # 多站点统计
            if "station_id" in gov_df.columns:
                st.markdown("---")
                st.subheader("📊 分站统计")
                station_stats = compute_station_statistics(gov_df)
                if not station_stats.empty:
                    st.dataframe(station_stats, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"读取数据失败: {e}")
    else:
        st.info(f"请先指定有效的数据文件路径（当前: `{gov_data_path}`）")

# ── 页脚 ──────────────────────────────────────────────────
st.markdown("---")
st.caption("水位数据异常检测系统 | LSTM Autoencoder + GIS | PyTorch + Streamlit | 汉江中下游流域研究区")
