"""Transform transaction features with a scaler fitted on training data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class Preprocessor:
    def __init__(self) -> None:
        self.scaler = RobustScaler()

    def _base(self, X: pd.DataFrame) -> pd.DataFrame:
        features = X.drop(columns="Time", errors="ignore").copy()
        features["Amount"] = np.log1p(features["Amount"])
        return features

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.scaler.fit(self._base(X)[["Amount"]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._base(X)
        features["Amount"] = self.scaler.transform(features[["Amount"]])[:, 0]
        return features
