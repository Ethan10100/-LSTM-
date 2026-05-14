"""
水位数据异常检测系统 - 异常检测模块（简化版）
"""
import numpy as np
from datetime import datetime

class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, threshold=2.0):
        self.threshold = threshold
    
    def detect(self, actual, predicted):
        """检测异常"""
        anomalies = []
        actual = np.array(actual[-len(predicted):])
        predicted = np.array(predicted)
        
        # 计算残差
        residuals = actual - predicted
        mean = np.mean(residuals)
        std = np.std(residuals)
        
        for i, (act, pred) in enumerate(zip(actual, predicted)):
            residual = act - pred
            z_score = abs((residual - mean) / std) if std != 0 else 0
            
            if z_score > self.threshold:
                anomaly_type = self._classify_anomaly(residual)
                anomalies.append({
                    'index': i,
                    'timestamp': datetime.now().isoformat(),
                    'actual_value': float(act),
                    'predicted_value': float(pred),
                    'anomaly_type': anomaly_type,
                    'confidence': float(min(z_score / (self.threshold * 2), 1.0)),
                    'z_score': float(z_score)
                })
        
        return anomalies
    
    def _classify_anomaly(self, residual):
        """分类异常类型"""
        if residual > 0:
            return 'spike'  # 突增异常
        else:
            return 'drop'   # 突降异常