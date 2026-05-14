# 水位数据异常检测系统（简化版）

基于 LSTM 的水位时序数据异常检测系统。

## 功能特性

- 水位数据采集与文件导入（CSV/Excel）
- 数据预处理：归一化、缺失值填充
- LSTM 深度学习模型训练与预测
- 水位异常自动检测与分类
- 数据可视化与异常日志记录

## 技术栈

- Python 3.8+
- Flask
- PyTorch
- Pandas / NumPy
- scikit-learn
- SQLite

## 项目结构

```
water_level_framework/
├── app/             # 主应用模块
│   ├── __init__.py  # 应用初始化
│   ├── database.py  # 数据库模型
│   ├── routes.py    # API路由
│   ├── model.py     # LSTM模型
│   ├── preprocessor.py  # 数据预处理
│   └── detector.py  # 异常检测
├── data/            # 数据文件
├── model_save/      # 模型保存
├── db/              # 数据库
├── test/            # 测试文件
├── requirements.txt # 依赖包
├── run.py           # 启动入口
└── README.md        # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行项目

```bash
python run.py
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/upload` | POST | 上传数据文件 |
| `/api/train` | POST | 训练模型 |
| `/api/detect` | POST | 检测异常 |
| `/api/data` | GET | 获取水位数据 |
| `/api/anomalies` | GET | 获取异常日志 |

## 使用示例

### 上传数据

```bash
curl -X POST -F "file=@water_level.csv" http://localhost:5002/api/upload
```

### 训练模型

```bash
curl -X POST -H "Content-Type: application/json" -d '{"station_id": "ST001"}' http://localhost:5002/api/train
```

### 检测异常

```bash
curl -X POST -H "Content-Type: application/json" -d '{"station_id": "ST001"}' http://localhost:5002/api/detect
```