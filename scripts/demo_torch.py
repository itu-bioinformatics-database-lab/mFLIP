import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Örnek veri seti oluşturma
# Giriş: 2 boyutlu, Çıkış: 1 boyutlu (y = x1 * 2 + x2 * 3 + 1)
np.random.seed(42)
X = np.random.rand(10000, 2).astype(np.float32)  # 100 örnek, 2 özellik
y = (X[:, 0] * 2 + X[:, 1] * 3 + 1).reshape(-1, 1)  # Basit bir lineer ilişki

# PyTorch tensörlerine dönüştürme
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Dataset ve DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# Basit bir sinir ağı tanımlama
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # 2 giriş, 16 gizli katman nöronu
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)  # 16 gizli katmandan 1 çıkış

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model, kayıp fonksiyonu ve optimizer tanımlama
model = SimpleNet()
model = model.to(device="cuda")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Eğitim döngüsü
epochs = 100
with tqdm(total=epochs, desc="Başlangıç") as pbar:
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            # Tahmin
            batch_X = batch_X.to(device="cuda")
            batch_y = batch_y.to(device="cuda")
            preds = model(batch_X)
            
            # Kayıp hesaplama
            loss = criterion(preds, batch_y)
            
            # Geri yayılım ve optimizasyon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        pbar.update(1)

# Modeli test etme
test_X = torch.tensor([[0.5, 0.2], [0.9, 0.8]], dtype=torch.float32)
test_X = test_X.to(device="cuda")

preds = model(test_X)
print("Test Tahminleri:", preds.detach().cpu().numpy())
