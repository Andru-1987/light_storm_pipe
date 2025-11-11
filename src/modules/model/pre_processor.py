import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
    Clase encargada de dividir, seleccionar solo columnas numéricas
    y escalar los datos de forma segura.
    """

    def __init__(self, df, target_col="target_encoded"):
        self.df = df
        self.target_col = target_col
        self.scaler = StandardScaler()

    def split_data(self):
        train_df = self.df[self.df["date"].dt.year < 2024].copy()
        test_df = self.df[self.df["date"].dt.year == 2024].copy()

        # Seleccionar features numéricas únicamente
        features = [
            c for c in self.df.select_dtypes(include=["number"]).columns
            if c not in [self.target_col]
        ]

        X_train, y_train = train_df[features], train_df[self.target_col]
        X_test, y_test = test_df[features], test_df[self.target_col]

        return X_train, X_test, y_train, y_test

    def scale(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
