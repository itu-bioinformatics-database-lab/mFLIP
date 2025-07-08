import os
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from deep_metabolitics.config import outputs_dir


def q_error(y_true, y_pred):
    """
    Q-error (quotient error) hesaplar.

    Parameters:
    -----------
    y_true : numpy.ndarray
        Gerçek değerler
    y_pred : numpy.ndarray
        Tahmin edilen değerler

    Returns:
    --------
    float or numpy.ndarray
        Her bir örnek için q-error değerleri veya ortalama q-error

    Notes:
    ------
    Q-error = max(tahmin/gerçek, gerçek/tahmin)
    """
    # Sıfıra bölme hatalarını önlemek için küçük bir epsilon değeri
    eps = 5

    # Girişleri numpy array'e çevir
    y_true = np.abs(np.asarray(y_true)) + eps
    y_pred = np.abs(np.asarray(y_pred)) + eps

    # Q-error hesaplama
    ratio1 = y_pred / y_true
    ratio2 = y_true / y_pred

    # Her örnek için maksimum oranı al
    q_errors = np.abs(np.maximum(ratio1, ratio2))

    return q_errors


# NumPy veya PyTorch tensörünü NumPy'ye çevir
def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


# Değerlendirme fonksiyonu (target isimlerine göre hesaplama yapar)
def evaluate_model(
    y_true, y_pred, target_names: List[str], dataset_type: str, scaler=None
):
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    if scaler is not None:
        print(f"{y_true.shape = }")
        print(f"{y_pred.shape = }")
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)

    # Genel istatistikler
    metrics = {
        f"{dataset_type}_{metric_name}_mixed": metric_value
        for metric_name, metric_value in compute_metrics(y_true, y_pred).items()
    }

    # _min içeren target'ları filtrele
    min_indices = [i for i, name in enumerate(target_names) if name.endswith("_min")]
    if min_indices:
        metrics.update(
            {
                f"{dataset_type}_{metric_name}_min": metric_value
                for metric_name, metric_value in compute_metrics(
                    y_true[:, min_indices], y_pred[:, min_indices]
                ).items()
            }
        )

    # _max içeren target'ları filtrele
    max_indices = [i for i, name in enumerate(target_names) if name.endswith("_max")]
    if max_indices:
        metrics.update(
            {
                f"{dataset_type}_{metric_name}_max": metric_value
                for metric_name, metric_value in compute_metrics(
                    y_true[:, max_indices], y_pred[:, max_indices]
                ).items()
            }
        )

    return metrics


# Metrik hesaplama fonksiyonu
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    q_errors = q_error(y_true, y_pred)
    q_err_mean = np.mean(q_errors)  # Ortalama Q-Error
    q_err_median = np.median(q_errors)  # Ortalama Q-Error
    r2_scores = r2_score(y_true, y_pred, multioutput="raw_values")
    r2_mean = np.mean(r2_scores)
    r2_median = np.median(r2_scores)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "q_err_mean": q_err_mean,
        "q_err_median": q_err_median,
        "r2_mean": r2_mean,
        "r2_median": r2_median,
        "mape": mape,
    }


class PerformanceMetricsUnseen:
    result_columns = [
        # "test_time_mixed",
        "test_rmse_mixed",
        "test_mae_mixed",
        "test_q_err_mean_mixed",
        "test_q_err_median_mixed",
        "test_r2_mean_mixed",
        "test_r2_median_mixed",
        "test_mape_mixed",
        # "test_time_min",
        "test_rmse_min",
        "test_mae_min",
        "test_q_err_mean_min",
        "test_q_err_median_min",
        "test_r2_mean_min",
        "test_r2_median_min",
        "test_mape_min",
        # "test_time_max",
        "test_rmse_max",
        "test_mae_max",
        "test_q_err_mean_max",
        "test_q_err_median_max",
        "test_r2_mean_max",
        "test_r2_median_max",
        "test_mape_max",
        # "validation_time_mixed",
        # "validation_rmse_mixed",
        # "validation_mae_mixed",
        # "validation_q_err_mean_mixed",
        # "validation_q_err_median_mixed",
        # "validation_r2_mean_mixed",
        # "validation_r2_median_mixed",
        # "validation_mape_mixed",
        # "validation_time_min",
        # "validation_rmse_min",
        # "validation_mae_min",
        # "validation_q_err_mean_min",
        # "validation_q_err_median_min",
        # "validation_r2_mean_min",
        # "validation_r2_median_min",
        # "validation_mape_min",
        # "validation_time_max",
        # "validation_rmse_max",
        # "validation_mae_max",
        # "validation_q_err_mean_max",
        # "validation_q_err_median_max",
        # "validation_r2_mean_max",
        # "validation_r2_median_max",
        # "validation_mape_max",
        # "train_time_mixed",
        # "train_rmse_mixed",
        # "train_mae_mixed",
        # "train_q_err_mean_mixed",
        # "train_q_err_median_mixed",
        # "train_r2_mean_mixed",
        # "train_r2_median_mixed",
        # "train_mape_mixed",
        # "train_time_min",
        # "train_rmse_min",
        # "train_mae_min",
        # "train_q_err_mean_min",
        # "train_q_err_median_min",
        # "train_r2_mean_min",
        # "train_r2_median_min",
        # "train_mape_min",
        # "train_time_max",
        # "train_rmse_max",
        # "train_mae_max",
        # "train_q_err_mean_max",
        # "train_q_err_median_max",
        # "train_r2_mean_max",
        # "train_r2_median_max",
        # "train_mape_max",
    ]

    def __init__(
        self,
        target_names,
        experience_name,
        ds_name,
        scaler=None,
    ):
        self.target_names = target_names
        self.experience_name = experience_name
        self.ds_name = ds_name
        self.scaler = scaler
        self.metrics = {}
        self.out_dir = outputs_dir / "unseen_metrics_cast_int" / self.experience_name
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def train_metric(self, y_true, y_pred):
        _metrics = evaluate_model(
            y_true=y_true,
            y_pred=y_pred,
            target_names=self.target_names,
            dataset_type="train",
            scaler=self.scaler,
        )
        self.metrics.update(_metrics)

    def validation_metric(self, y_true, y_pred):
        _metrics = evaluate_model(
            y_true=y_true,
            y_pred=y_pred,
            target_names=self.target_names,
            dataset_type="validation",
            scaler=self.scaler,
        )
        self.metrics.update(_metrics)

    def test_metric(self, y_true, y_pred):
        _metrics = evaluate_model(
            y_true=y_true,
            y_pred=y_pred,
            target_names=self.target_names,
            dataset_type="test",
            scaler=self.scaler,
        )
        self.metrics.update(_metrics)

    def complete(self):
        self.metrics_df = pd.DataFrame([self.metrics])
        self.metrics_df = self.metrics_df[self.result_columns]
        self.fpath = os.path.join(
            self.out_dir, f"performance_metrics_{self.ds_name}.csv"
        )
        self.metrics_df.to_csv(self.fpath)
        print(f"{self.fpath = }")
        return self.metrics_df
