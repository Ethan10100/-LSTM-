"""
水位数据异常检测系统 - 数据预处理模块（简化版）
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def normalize(self, data):
        """归一化数据"""
        data = np.array(data).reshape(-1, 1)
        self.scaler.fit(data)
        return self.scaler.transform(data).flatten()
    
    def denormalize(self, data):
        """反归一化数据"""
        data = np.array(data).reshape(-1, 1)
        return self.scaler.inverse_transform(data).flatten()
    
    def fill_missing_values(self, data):
        """填充缺失值（简单插值法）"""
        data = np.array(data)
        mask = np.isnan(data)
        if np.any(mask):
            x = np.arange(len(data))
            data[mask] = np.interp(x[mask], x[~mask], data[~mask])
        return data
    
    def create_sequences(self, data, seq_length=24):
        """创建时间序列数据"""
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
        return np.array(X)