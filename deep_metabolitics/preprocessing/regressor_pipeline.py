import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# 1. Custom Dataset Class
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. PyTorch Model
class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# 3. Training Function
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

# 4. Sklearn Pipeline
class PyTorchRegressor:
    def __init__(self, input_dim, output_dim, epochs=50, lr=0.001):
        self.model = RegressionModel(input_dim, output_dim)
        self.epochs = epochs
        self.lr = lr
    
    def fit(self, X, y):
        dataset = RegressionDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.model = train_model(self.model, loader, loader, self.epochs, self.lr)
        return self
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).numpy()

# 5. Pipeline Kullanımı
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', VarianceThreshold(threshold=0.01)),
    ('regressor', PyTorchRegressor(input_dim=312, output_dim=98, epochs=50, lr=0.001))
])

# 6. Train Pipeline
# train_df, val_df, test_df dataframe'leri mevcut olmalı
X_train, y_train = train_df.iloc[:, :-98], train_df.iloc[:, -98:]
pipeline.fit(X_train, y_train)

# 7. Prediction
X_test = test_df.iloc[:, :-98]
y_pred = pipeline.predict(X_test)
