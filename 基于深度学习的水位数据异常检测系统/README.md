# 基于深度学习的水位数据异常检测系统

本项目演示一个基于深度学习的水位数据异常检测系统，使用 LSTM 自编码器检测水位时间序列中的异常点。

## 项目结构

- `requirements.txt`：依赖列表
- `src/data_utils.py`：数据加载与预处理工具
- `src/model.py`：LSTM 自编码器模型定义
- `src/train.py`：模型训练脚本
- `src/detect.py`：异常检测脚本
- `src/config.py`：配置参数

## 使用说明

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 准备数据

将水位时间序列数据保存为 CSV 文件，至少包含一列 `water_level`。示例文件：`data/water_level.csv`

3. 训练模型

```bash
python src/train.py --data-path data/water_level.csv --model-path models/lstm_autoencoder.pth
```

4. 执行异常检测

```bash
python src/detect.py --data-path data/water_level.csv --model-path models/lstm_autoencoder.pth
```

## 算法说明

- 使用 LSTM 自编码器对水位序列建模
- 将正常序列重构误差作为异常评分
- 重构误差超过阈值的点被判定为异常

## 目录建议

- `data/`：放置原始数据集
- `models/`：保存训练好的模型
- `results/`：保存检测结果和可视化图表
