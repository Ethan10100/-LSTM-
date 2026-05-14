水位数据异常检测系统（基于 LSTM）
本项目为课程大作业，基于深度学习 LSTM 模型实现水位时序数据的异常检测，包含数据采集、预处理、模型训练、异常识别与可视化功能。
📌 项目功能
水位数据采集与文件导入（CSV/Excel）
数据预处理：缺失值填充、降噪、归一化
LSTM 深度学习模型训练与预测
水位异常自动检测与分类
数据可视化与异常日志记录
🛠️ 技术栈
Python 3.8+
Flask
PyTorch
Pandas / NumPy
Matplotlib
SQLite
📂 项目结构
plaintext
water_level_anomaly_detection/
├── app/              # 主应用模块
├── data/             # 水位数据文件
├── model_save/       # 模型保存
├── db/               # 数据库
├── test/             # 测试文件
├── requirements.txt  # 依赖包
├── run.py            # 项目启动入口
└── README.md
🚀 运行说明
1. 克隆仓库
bash
运行
git clone https://github.com/Ethan10100/-LSTM-.git
cd -LSTM-
2. 安装依赖
bash
运行
pip install -r requirements.txt
3. 启动项目
bash
运行
python run.py
📅 开发进度
表格
周次	任务	状态
第 1 周	项目框架搭建、数据采集模块	✅ 已完成
第 2 周	数据预处理、LSTM 模型开发	🔄 开发中
第 3 周	异常检测、接口开发	⏳ 待开发
第 4 周	可视化、日志、系统联调	⏳ 待开发
第 5 周	测试、验收、文档	⏳ 待开发
👥 小组成员
卓宇聪（负责人）：项目框架、GitHub 管理、系统配置
童宇昊：算法开发、数据预处理、LSTM 模型
朱辰韬：异常检测、接口、可视化
📄 说明
本项目为浙江水利水电学院课程大作业，仅用于教学与学习目的，未经许可不得商用。

