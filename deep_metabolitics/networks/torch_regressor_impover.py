import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from deep_metabolitics.networks.metabolite_impoverfcnn import MetaboliteImpFCNN
from deep_metabolitics.utils.trainer_pm import predict_own_dnn, train_own_dnn



class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=5835, hidden_dims=[2048, 1024, 512, 256], dropout_rate=0.2, num_residual_blocks=2, epochs=10, batch_size=32, device='cuda'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.num_residual_blocks = num_residual_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.model = MetaboliteImpFCNN(
            input_dim=self.input_dim,
            output_dim=1,
            hidden_dims=self.hidden_dims,
            num_residual_blocks=self.num_residual_blocks,
            dropout_rate=self.dropout_rate,
        )


    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)

        self.model, _ = train_own_dnn(
            train_dataset=dataset,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            learning_rate=0.00001,
            weight_decay=0.01,
            epochs=self.epochs,
        )

        return self

    def predict(self, X):
        self.model.to(self.device)
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X)
            preds = preds["pathways_pred"]

        return preds.squeeze().cpu().numpy()
