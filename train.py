"""
水位数据异常检测系统 - 模型训练脚本（第二步）
完整流程：数据加载 → 预处理 → LSTM模型训练 → 评估 → 保存
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

from app.preprocessor import DataPreprocessor
from app.model import LSTMNetwork, LSTMTrainer, ModelManager

def generate_sample_data(n_samples=1000, noise_level=0.1):
    """
    生成模拟水位数据（用于演示）
    :param n_samples: 样本数量
    :param noise_level: 噪声水平
    :return: DataFrame
    """
    np.random.seed(42)
    
    # 生成时间序列
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    # 生成基础水位（正弦波 + 趋势 + 噪声）
    t = np.arange(n_samples)
    base_level = 50 + 10 * np.sin(2 * np.pi * t / 24)  # 日周期变化
    trend = 0.01 * t  # 缓慢上升趋势
    noise = np.random.normal(0, noise_level * 10, n_samples)
    
    water_level = base_level + trend + noise
    
    # 添加一些异常值
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    water_level[anomaly_indices] += np.random.choice([-1, 1], size=len(anomaly_indices)) * np.random.uniform(15, 25, size=len(anomaly_indices))
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'water_level': water_level,
        'temperature': 20 + 5 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1, n_samples),
        'rainfall': np.random.exponential(2, n_samples) * (np.random.random(n_samples) > 0.8)
    })
    
    return df

def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史曲线"""
    plt.figure(figsize=(15, 5))
    
    # Loss曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Loss Curve', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['train_mae'], label='Train MAE', linewidth=2)
    plt.plot(history['val_mae'], label='Val MAE', linewidth=2)
    plt.title('MAE Curve', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 预测对比
    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.8, 'Training Complete!', fontsize=16, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Final Train Loss: {history['train_loss'][-1]:.6f}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Final Val Loss: {history['val_loss'][-1]:.6f}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Final Train MAE: {history['train_mae'][-1]:.6f}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f"Final Val MAE: {history['val_mae'][-1]:.6f}", fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练历史图已保存: {save_path}")
    plt.close()

def plot_predictions(y_true, y_pred, save_path='predictions.png'):
    """绘制预测结果对比"""
    plt.figure(figsize=(14, 6))
    
    # 时间序列对比
    plt.subplot(1, 2, 1)
    plt.plot(y_true[:200], label='True', linewidth=2, alpha=0.8)
    plt.plot(y_pred[:200], label='Predicted', linewidth=2, alpha=0.8)
    plt.title('Prediction vs True (First 200 samples)', fontsize=12)
    plt.xlabel('Time Step')
    plt.ylabel('Water Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 散点图
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2, label='Perfect Prediction')
    plt.title('Prediction Scatter Plot', fontsize=12)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测对比图已保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 70)
    print("水位数据异常检测系统 - 第二步：数据预处理 + LSTM模型训练")
    print("=" * 70)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建目录
    os.makedirs('model_save', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # ==================== 1. 数据加载 ====================
    print("\n" + "-" * 70)
    print("步骤 1: 数据加载")
    print("-" * 70)
    
    # 检查是否有真实数据文件
    data_file = 'data/water_level_data.csv'
    if os.path.exists(data_file):
        print(f"加载真实数据: {data_file}")
        df = pd.read_csv(data_file)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print("未找到真实数据，生成模拟数据...")
        df = generate_sample_data(n_samples=2000)
        df.to_csv(data_file, index=False)
        print(f"模拟数据已保存: {data_file}")
    
    print(f"数据形状: {df.shape}")
    print(f"数据列: {list(df.columns)}")
    print(f"\n数据预览:")
    print(df.head())
    
    # ==================== 2. 数据预处理 ====================
    print("\n" + "-" * 70)
    print("步骤 2: 数据预处理")
    print("-" * 70)
    
    # 初始化预处理器
    preprocessor = DataPreprocessor(scaler_type='minmax')
    
    # 完整预处理流程
    feature_cols = ['water_level', 'temperature', 'rainfall'] if 'temperature' in df.columns else None
    
    processed_data = preprocessor.prepare_data(
        df=df,
        target_col='water_level',
        feature_cols=feature_cols,
        seq_length=24,      # 使用过去24小时预测下一小时
        test_ratio=0.2,
        val_ratio=0.1
    )
    
    print(f"\n预处理完成:")
    print(f"  训练集: {processed_data['X_train'].shape}")
    print(f"  验证集: {processed_data['X_val'].shape}")
    print(f"  测试集: {processed_data['X_test'].shape}")
    print(f"  序列长度: {processed_data['seq_length']}")
    
    # ==================== 3. LSTM模型训练 ====================
    print("\n" + "-" * 70)
    print("步骤 3: LSTM模型训练")
    print("-" * 70)
    
    # 创建LSTM模型
    input_size = processed_data['X_train'].shape[2]  # 特征维度
    model = LSTMNetwork(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    )
    
    print(f"\n模型结构:")
    print(model)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = LSTMTrainer(
        model=model,
        device=device,
        learning_rate=0.001
    )
    
    # 训练模型
    history = trainer.train(
        train_data=(processed_data['X_train'], processed_data['y_train']),
        val_data=(processed_data['X_val'], processed_data['y_val']),
        epochs=50,
        batch_size=32,
        patience=10
    )
    
    # 绘制训练历史
    plot_training_history(history, save_path='model_save/training_history.png')
    
    # ==================== 4. 模型评估 ====================
    print("\n" + "-" * 70)
    print("步骤 4: 模型评估")
    print("-" * 70)
    
    # 加载最佳模型
    trainer.load_checkpoint('best_model.pth')
    
    # 评估模型
    metrics = trainer.evaluate(
        processed_data['X_test'],
        processed_data['y_test']
    )
    
    # 绘制预测结果
    y_pred = trainer.predict(processed_data['X_test'])
    y_true = processed_data['y_test']
    plot_predictions(y_true, y_pred, save_path='model_save/predictions.png')
    
    # ==================== 5. 保存模型信息 ====================
    print("\n" + "-" * 70)
    print("步骤 5: 保存模型信息")
    print("-" * 70)
    
    # 保存模型配置
    model_config = {
        'input_size': input_size,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 1,
        'dropout': 0.2,
        'seq_length': processed_data['seq_length'],
        'scaler_type': 'minmax'
    }
    
    with open('model_save/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # 保存训练信息
    training_info = {
        'model_name': 'LSTM_WaterLevel_Predictor',
        'created_at': datetime.now().isoformat(),
        'device': str(device),
        'metrics': metrics,
        'model_config': model_config,
        'training_params': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10
        },
        'data_info': {
            'total_samples': len(df),
            'train_samples': len(processed_data['X_train']),
            'val_samples': len(processed_data['X_val']),
            'test_samples': len(processed_data['X_test']),
            'features': feature_cols if feature_cols else ['water_level']
        }
    }
    
    with open('model_save/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n模型配置已保存: model_save/model_config.json")
    print(f"训练信息已保存: model_save/training_info.json")
    
    # 保存预处理器
    import pickle
    with open('model_save/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"预处理器已保存: model_save/preprocessor.pkl")
    
    # ==================== 6. 总结 ====================
    print("\n" + "=" * 70)
    print("训练完成总结")
    print("=" * 70)
    print(f"\n模型文件:")
    print(f"  - model_save/best_model.pth (最佳模型权重)")
    print(f"  - model_save/model_config.json (模型配置)")
    print(f"  - model_save/training_info.json (训练信息)")
    print(f"  - model_save/preprocessor.pkl (预处理器)")
    print(f"  - model_save/training_history.png (训练曲线)")
    print(f"  - model_save/predictions.png (预测对比)")
    print(f"\n评估指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    print("\n" + "=" * 70)
    print("第二步完成！可以进行异常检测了。")
    print("=" * 70)

if __name__ == '__main__':
    main()