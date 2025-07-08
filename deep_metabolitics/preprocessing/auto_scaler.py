from calendar import c

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted


class AutoScaler(BaseEstimator, TransformerMixin):
    """
    Automatically selects and applies the most appropriate scaler for each feature.

    Parameters:
    -----------
    threshold_skewness : float, default=1.0
        Threshold for determining if a feature is highly skewed
    threshold_outliers : float, default=2.0
        Z-score threshold for determining if a feature has significant outliers
    """

    def __init__(
        self, threshold_skewness=1.0, threshold_outliers=2.0, threshold_normality=0.05
    ):
        self.threshold_skewness = threshold_skewness
        self.threshold_outliers = threshold_outliers
        self.threshold_normality = threshold_normality
        self.scalers_ = {}
        self.feature_names_ = None

    # def _detect_scaler(self, X):
    #     """Determines appropriate scaler for each feature."""
    #     scalers = {}

    #     for column in X.columns:
    #         data = X[column].values

    #         # Calculate metrics
    #         skewness = abs(stats.skew(data))
    #         z_scores = np.abs(stats.zscore(data))
    #         has_outliers = np.any(z_scores > self.threshold_outliers)

    #         # Decision logic for scaler selection
    #         if has_outliers:
    #             scalers[column] = RobustScaler()
    #         elif skewness > self.threshold_skewness:
    #             scalers[column] = MinMaxScaler()
    #         else:
    #             scalers[column] = StandardScaler()

    #     return scalers

    def _detect_scaler(self, X):
        """Determines appropriate scaler for each feature."""
        scalers = {}

        for column in X.columns:
            data = X[column].values

            # Calculate metrics
            skewness = abs(stats.skew(data))
            z_scores = np.abs(stats.zscore(data))
            has_outliers = np.any(z_scores > self.threshold_outliers)

            # Shapiro-Wilk testi için örnek boyutu kontrolü
            if len(data) < 3:
                _, normality_p_value = 0, 0
            else:
                _, normality_p_value = stats.shapiro(data)

            # Geliştirilmiş karar mantığı
            if has_outliers:
                if skewness > self.threshold_skewness:
                    scalers[column] = QuantileTransformer(output_distribution="normal")
                else:
                    scalers[column] = StandardScaler()
            elif normality_p_value < self.threshold_normality:
                if skewness > self.threshold_skewness:
                    scalers[column] = PowerTransformer(method="yeo-johnson")
                else:
                    scalers[column] = MinMaxScaler()
            else:
                scalers[column] = StandardScaler()
            print(column, scalers[column])
        return scalers

    def fit(self, X, y=None):
        """
        Fit the scaler for each feature.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to be transformed
        y : None
            Ignored

        Returns:
        --------
        self : object
            Returns self
        """
        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.feature_names_ = X.columns.tolist()
        self.scalers_ = self._detect_scaler(X)

        # Fit each scaler
        for column, scaler in self.scalers_.items():
            scaler.fit(X[[column]])

        return self

    def transform(self, X):
        """
        Transform the data using the fitted scalers.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to be transformed

        Returns:
        --------
        pandas DataFrame
            Transformed data
        """
        check_is_fitted(self, ["scalers_", "feature_names_"])

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if not all(col in X.columns for col in self.feature_names_):
            raise ValueError("Transform data missing columns from fitting")

        result = np.zeros(X.shape)
        for i, (column, scaler) in enumerate(self.scalers_.items()):
            result[:, i] = scaler.transform(X[column].values.reshape(-1, 1)).ravel()

        return result

    def inverse_transform(self, X):
        """
        Scale back the data to the original representation.

        Parameters:
        -----------
        X : pandas DataFrame
            The transformed data

        Returns:
        --------
        pandas DataFrame
            The inverse transformed data
        """
        check_is_fitted(self, ["scalers_", "feature_names_"])

        # if not isinstance(X, pd.DataFrame):
        #     raise ValueError("Input must be a pandas DataFrame")

        # if not all(col in X.columns for col in self.feature_names_):
        #     raise ValueError("Inverse transform data missing columns from fitting")

        result = np.zeros(X.shape)
        for i, (column, scaler) in enumerate(self.scalers_.items()):
            result[:, i] = scaler.inverse_transform(X[:, i].reshape(-1, 1)).ravel()

        # Preserve the original column order
        return result

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to be transformed
        y : None
            Ignored

        Returns:
        --------
        pandas DataFrame
            Transformed data
        """
        return self.fit(X).transform(X)

    def get_feature_scalers(self):
        """
        Returns a dictionary of feature names and their corresponding scalers.

        Returns:
        --------
        dict
            Dictionary with feature names as keys and scaler types as values
        """
        check_is_fitted(self, ["scalers_"])
        return {
            feature: type(scaler).__name__ for feature, scaler in self.scalers_.items()
        }
