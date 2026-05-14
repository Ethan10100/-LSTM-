"""
水位数据异常检测系统 - 应用初始化（简化版）
"""
from flask import Flask
from flask_cors import CORS
from app.routes import api_bp
from app.database import init_db
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'water_level_secret_key'
    
    # 配置数据库路径
    db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'water_level.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # 初始化数据库
    init_db(app)
    
    # 注册蓝图
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # 启用CORS
    CORS(app)
    
    return app