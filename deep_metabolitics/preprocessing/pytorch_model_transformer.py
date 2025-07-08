import logging
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin


class PyTorchModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        batch_size: int = 32,
        input_transform: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.logger = logger or logging.getLogger(__name__)

    def fit(self, X, y=None):
        self.logger.info("PyTorchModelTransformer fit called")
        return self

    def transform(self, X):
        self.logger.info(f"Transforming data of shape {X.shape}")
        self.model.eval()
        with torch.no_grad():
            all_predictions = self.model(torch.tensor(X, device="cuda"))

        all_predictions = all_predictions["pathways_pred"].cpu().numpy()

        return all_predictions
