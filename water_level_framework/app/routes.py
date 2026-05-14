"""
水位数据异常检测系统 - API路由（简化版）
"""
from flask import Blueprint, request, jsonify
import os
import numpy as np
import pandas as pd
from app.database import db, WaterLevelData, AnomalyLog
from app.preprocessor import DataPreprocessor
from app.model import LSTMModelHandler
from app.detector import AnomalyDetector
from datetime import datetime

api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@api_bp.route('/upload', methods=['POST'])
def upload_data():
    """上传水位数据"""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': '请上传文件'}), 400
        
        # 保存文件
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, file.filename)
        file.save(file_path)
        
        # 读取并保存到数据库
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            data = WaterLevelData(
                timestamp=pd.to_datetime(row['timestamp']),
                water_level=row['water_level'],
                station_id=row.get('station_id', 'ST001')
            )
            db.session.add(data)
        db.session.commit()
        
        return jsonify({'message': '数据上传成功', 'count': len(df)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/train', methods=['POST'])
def train_model():
    """训练模型"""
    try:
        station_id = request.json.get('station_id', 'ST001')
        data = WaterLevelData.query.filter_by(station_id=station_id).all()
        
        if len(data) < 100:
            return jsonify({'error': '数据量不足，至少需要100条数据'}), 400
        
        # 准备数据
        values = [d.water_level for d in data]
        
        # 预处理
        preprocessor = DataPreprocessor()
        normalized_data = preprocessor.normalize(values)
        
        # 训练模型
        model_handler = LSTMModelHandler()
        model_handler.train(normalized_data)
        model_handler.save_model()
        
        return jsonify({'message': '模型训练完成'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/detect', methods=['POST'])
def detect_anomaly():
    """检测异常"""
    try:
        station_id = request.json.get('station_id', 'ST001')
        data = WaterLevelData.query.filter_by(station_id=station_id).order_by(WaterLevelData.timestamp).all()
        
        if len(data) < 24:
            return jsonify({'error': '数据量不足'}), 400
        
        values = [d.water_level for d in data]
        
        # 加载模型并预测
        model_handler = LSTMModelHandler()
        model_handler.load_model()
        
        preprocessor = DataPreprocessor()
        normalized_data = preprocessor.normalize(values)
        
        predictions = model_handler.predict(normalized_data)
        predictions = preprocessor.denormalize(predictions)
        
        # 检测异常
        detector = AnomalyDetector()
        anomalies = detector.detect(values, predictions)
        
        # 保存异常日志
        for anomaly in anomalies:
            log = AnomalyLog(
                station_id=station_id,
                timestamp=anomaly['timestamp'],
                actual_value=anomaly['actual_value'],
                predicted_value=anomaly['predicted_value'],
                anomaly_type=anomaly['anomaly_type'],
                confidence=anomaly['confidence']
            )
            db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'message': '异常检测完成',
            'total_records': len(data),
            'anomaly_count': len(anomalies),
            'anomalies': anomalies
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/data', methods=['GET'])
def get_data():
    """获取水位数据"""
    try:
        station_id = request.args.get('station_id', 'ST001')
        data = WaterLevelData.query.filter_by(station_id=station_id).order_by(WaterLevelData.timestamp).all()
        return jsonify([d.to_dict() for d in data]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/anomalies', methods=['GET'])
def get_anomalies():
    """获取异常日志"""
    try:
        station_id = request.args.get('station_id')
        query = AnomalyLog.query
        if station_id:
            query = query.filter_by(station_id=station_id)
        data = query.order_by(AnomalyLog.timestamp.desc()).all()
        return jsonify([d.to_dict() for d in data]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500