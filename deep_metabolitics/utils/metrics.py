import numpy as np
import torch


def get_mse(A: np.ndarray, B: np.ndarray, rowwise: bool = True) -> np.ndarray:
    """
    Calculate the Mean Squared Error (MSE) between two matrices A and B.

    Parameters:
        - A (numpy.ndarray): First matrix.
        - B (numpy.ndarray): Second matrix.
        - rowwise (bool): If True, calculate row-wise MSE; if False, calculate column-wise MSE.

    Returns:
        numpy.ndarray: Array of MSE values.
    """
    ax = 0 if rowwise else 1
    mse = np.mean((A - B) ** 2, axis=ax)

    return mse


def matrix_mse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between correlation matrices of two datasets.

    Parameters:
        - x (numpy.ndarray): First dataset.
        - y (numpy.ndarray): Second dataset.

    Returns:
        float: Mean Squared Error (MSE) between the correlation matrices.
    """
    corr_x = np.corrcoef(x, rowvar=False).flatten()
    corr_y = np.corrcoef(y, rowvar=False).flatten()
    mses = get_mse(corr_x, corr_y)

    return np.mean(mses)


def R2(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the R-squared (coefficient of determination) between true and predicted values.

    Parameters:
    - y_pred (torch.Tensor): Predicted target values.
    - y_true (torch.Tensor): True target values.

    Returns:
    - torch.Tensor: R-squared value.
    """
    assert y_pred.shape == y_true.shape, "Shape mismatch between y_pred and y_true"

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)

    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    mean_true = torch.mean(y_true)
    tss = torch.sum((y_true - mean_true) ** 2)  # total sum of squares (TSS)
    rss = torch.sum((y_true - y_pred) ** 2)  # residual sum of squares (RSS)

    r_squared = 1 - (rss / tss)

    return r_squared


def MSE(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the MSE (mean squared error) between true and predicted values.

    Parameters:
    - y_pred (torch.Tensor): Predicted target values.
    - y_true (torch.Tensor): True target values.

    Returns:
    - torch.Tensor: MSE value.
    """
    return torch.mean((y_pred - y_true) ** 2)


def RMSELoss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Root Mean Squared Error (RMSE) loss between predicted (yhat) and target (y) values.

    Parameters:
    - yhat (torch.Tensor): Predicted values.
    - y (torch.Tensor): Target (ground truth) values.

    Returns:
    - torch.Tensor: Root Mean Squared Error (RMSE) loss.
    """
    assert yhat.shape == y.shape, "Shape mismatch between yhat and y"

    # Add a small epsilon = 1e-8 to avoid division by zero as
    # in backpropagation it will result in nans!
    return torch.sqrt(torch.mean((yhat - y) ** 2) + 1e-8)


def NRMSELoss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE) between two tensors.

    Parameters:
    - yhat (torch.Tensor): Predicted tensor.
    - y (torch.Tensor): Target tensor.

    Returns:
    - torch.Tensor: Normalized Root Mean Squared Error (NRMSE) value.
    """
    assert yhat.shape == y.shape, "Shape mismatch between yhat and y"

    rmse = torch.sqrt(torch.mean((yhat - y) ** 2))
    print(f"{rmse = }")
    y_range = torch.max(y, dim=0)[0] - torch.min(y, dim=0)[0]
    print(f"{y_range = }")
    # Add a small epsilon to avoid division by zero
    normalized_rmse = rmse / (y_range + 1e-8)

    return normalized_rmse


def MAE(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the Mean Absolute Error (coefficient of determination) between true and predicted values.

    Parameters:
    - y_pred (torch.Tensor): Predicted target values.
    - y_true (torch.Tensor): True target values.

    Returns:
    - torch.Tensor: Mean Absolute Error value.
    """
    assert y_pred.shape == y_true.shape, "Shape mismatch between y_pred and y_true"

    error = torch.nn.functional.l1_loss(y_pred, y_true)

    # MAE = torch.nn.L1Loss()
    # error = MAE(y_pred, y_true)

    return error


def calculate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor):
    mse = MSE(y_pred=y_pred, y_true=y_true)
    mae = MAE(y_pred=y_pred, y_true=y_true)
    r2 = R2(y_pred=y_pred, y_true=y_true)
    rmse = RMSELoss(yhat=y_pred, y=y_true)
    return mse, mae, r2, rmse
