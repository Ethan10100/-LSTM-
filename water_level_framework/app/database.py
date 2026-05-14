"""
水位数据异常检测系统 - 数据库模型（简化版）
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class WaterLevelData(db.Model):
    """水位数据表"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    water_level = db.Column(db.Float, nullable=False)
    station_id = db.Column(db.String(50), nullable=False)
    is_anomaly = db.Column(db.Boolean, default=False)
    anomaly_type = db.Column(db.String(50), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'water_level': self.water_level,
            'station_id': self.station_id,
            'is_anomaly': self.is_anomaly,
            'anomaly_type': self.anomaly_type
        }

class AnomalyLog(db.Model):
    """异常日志表"""
    id = db.Column(db.Integer, primary_key=True)
    station_id = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    actual_value = db.Column(db.Float, nullable=False)
    predicted_value = db.Column(db.Float, nullable=False)
    anomaly_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'station_id': self.station_id,
            'timestamp': self.timestamp.isoformat(),
            'actual_value': self.actual_value,
            'predicted_value': self.predicted_value,
            'anomaly_type': self.anomaly_type,
            'confidence': self.confidence
        }

def init_db(app):
    """初始化数据库"""
    db.init_app(app)
    with app.app_context():
        db.create_all()