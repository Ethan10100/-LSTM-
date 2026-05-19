# 水位数据异常检测系统 - 第二步完成版

## 📋 项目概述

基于LSTM深度学习模型的水位时序数据异常检测系统。

## ✅ 第二步完成内容

### 1. 数据预处理模块 (app/preprocessor.py)
- ✅ **缺失值处理**: 支持线性插值、前向填充、均值/中位数填充
- ✅ **异常值处理**: Z-score方法和IQR方法
- ✅ **数据归一化**: MinMaxScaler和StandardScaler
- ✅ **时序数据构造**: 创建LSTM输入序列 (batch_size, seq_length, features)
- ✅ **数据集划分**: 自动划分训练集/验证集/测试集

### 2. LSTM模型模块 (app/model.py)
- ✅ **LSTM网络搭建**:
  - 输入层: 支持多特征输入
  - LSTM层: 2层LSTM，隐藏层大小64
  - Dropout层: 防止过拟合
  - 全连接层: 输出预测值
- ✅ **模型训练脚本**:
  - 支持早停机制
  - 学习率自适应调整
  - 训练历史记录
  - 最佳模型自动保存
- ✅ **模型评估**: MSE, RMSE, MAE, MAPE指标

### 3. 训练脚本 (train.py)
- ✅ 完整训练流程
- ✅ 模拟数据生成（用于演示）
- ✅ 训练可视化
- ✅ 模型信息保存

## 📁 项目结构

```
water_level_framework/
├── app/
│   ├── __init__.py              # 应用初始化
│   ├── database.py              # 数据库模型
│   ├── routes.py                # API路由（完善版）
│   ├── preprocessor.py          # 数据预处理（第二步完善）
│   ├── model.py                 # LSTM模型（第二步完善）
│   └── detector.py              # 异常检测
├── data/
│   └── water_level_data.csv     # 示例数据
├── model_save/                  # 模型保存目录
│   ├── best_model.pth           # 最佳模型权重
│   ├── model_config.json        # 模型配置
│   ├── training_info.json       # 训练信息
│   ├── preprocessor.pkl         # 预处理器
│   ├── training_history.png     # 训练曲线
│   └── predictions.png          # 预测对比图
├── db/                          # 数据库目录
├── train.py                     # 训练脚本（第二步新增）
├── run.py                       # 服务启动入口
├── requirements.txt             # 依赖配置
└── README_STEP2.md              # 本文件
```

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行训练
```bash
python train.py
```

### 启动API服务
```bash
python run.py
```

## 📊 模型性能

| 指标 | 值 |
|------|-----|
| MSE | 0.008439 |
| RMSE | 0.091863 |
| MAE | 0.050014 |
| MAPE | 11.96% |

## 🌐 API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/upload` | POST | 上传数据文件 |
| `/api/train` | POST | 训练模型 |
| `/api/predict` | POST | 预测水位 |
| `/api/detect` | POST | 检测异常 |
| `/api/data` | GET | 获取水位数据 |
| `/api/anomalies` | GET | 获取异常日志 |
| `/api/model/info` | GET | 获取模型信息 |

## 📝 使用示例

### 训练模型
```bash
curl -X POST http://localhost:5002/api/train \
  -H "Content-Type: application/json" \
  -d '{"station_id": "ST001", "epochs": 50}'
```

### 预测水位
```bash
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"station_id": "ST001"}'
```

### 检测异常
```bash
curl -X POST http://localhost:5002/api/detect \
  -H "Content-Type: application/json" \
  -d '{"station_id": "ST001", "threshold": 2.0}'
```

## 🔧 技术栈

- Python 3.8+
- PyTorch 2.0+
- Flask
- Pandas / NumPy
- scikit-learn
- SQLite

## 📈 下一步计划

- 第三步：完善异常检测算法
- 第四步：添加可视化界面
- 第五步：系统部署与测试
