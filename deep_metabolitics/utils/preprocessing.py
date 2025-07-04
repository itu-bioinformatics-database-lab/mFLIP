import numpy as np


def own_log_scaler(X):
    scaled_X = (X / X.abs()) * np.log1p(X.abs())
    return scaled_X


def own_inverse_log_scaler(scaled_X):
    inversed_X = (scaled_X / scaled_X.abs()) * np.expm1(scaled_X.abs())
    return inversed_X
