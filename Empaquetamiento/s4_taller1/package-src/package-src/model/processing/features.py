from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper with safe fallback."""

    def __init__(self, variables: List[str], mappings: dict):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            # Map and handle unseen labels by preserving original value
            X[feature] = X[feature].map(self.mappings).fillna(X[feature])
        return X

