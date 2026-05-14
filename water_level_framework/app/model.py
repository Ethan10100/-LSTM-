"""
水位数据异常检测系统 - LSTM模型模块（简化版）
"""
import torch
import torch.nn as nn
import numpy as np
import os

class LSTMModel(nn.Module):
    """LSTM模型"""
    
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModelHandler:
    """模型处理器"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self):
        """构建模型"""
        self.model = LSTMModel().to(self.device)
    
    def train(self, data, epochs=50, seq_length=24, lr=0.001):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        # 创建序列
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        X = np.array(X).reshape(-1, seq_length, 1)
        y = np.array(y).reshape(-1, 1)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    def predict(self, data, seq_length=24):
        """预测"""
        if self.model is None:
            self.load_model()
        
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
        
        X = np.array(X).reshape(-1, seq_length, 1)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        return outputs.cpu().numpy().flatten()
    
    def save_model(self):
        """保存模型"""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_save')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'lstm_model.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f'模型已保存到: {model_path}')
    
    def load_model(self):
        """加载模型"""
        if self.model is None:
            self.build_model()
        
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_save', 'lstm_model.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f'模型已从 {model_path} 加载')
        else:
            print('未找到预训练模型，使用新模型')