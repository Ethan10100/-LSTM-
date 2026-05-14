"""
水位数据异常检测系统 - 主入口文件（简化版）
"""
from app import create_app

app = create_app()

if __name__ == '__main__':
    print("🚀 启动水位数据异常检测系统（简化版）")
    print("📡 API服务运行在: http://localhost:5002")
    print("🔧 可用接口:")
    print("   - GET  /api/health    - 健康检查")
    print("   - POST /api/upload    - 上传数据文件")
    print("   - POST /api/train     - 训练模型")
    print("   - POST /api/detect    - 检测异常")
    print("   - GET  /api/data      - 获取水位数据")
    print("   - GET  /api/anomalies - 获取异常日志")
    print("\n按 Ctrl+C 停止服务")
    
    app.run(host='0.0.0.0', port=5002, debug=True)