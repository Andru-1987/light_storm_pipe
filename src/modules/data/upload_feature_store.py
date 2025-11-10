import os
import pandas as pd
from datetime import datetime

class FeatureStoreManager:
    """
    Clase para gestionar el almacenamiento y carga de features preprocesados
    """

    def __init__(self, root_path: str):
        self.root_path = root_path
        self.preprocessed_dir = os.path.join(root_path, "data", "preprocessed")
        os.makedirs(self.preprocessed_dir, exist_ok=True)

    def _generate_filename(self, name: str = "forex_features", versioned: bool = True) -> str:
        """
        Genera el nombre de archivo (con versión timestamp opcional)
        """
        if versioned:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.csv"
        else:
            filename = f"{name}.csv"
        return os.path.join(self.preprocessed_dir, filename)

    def save_features(self, df: pd.DataFrame, name: str = "forex_features", versioned: bool = True) -> str:
        """
        Guarda el DataFrame de features como CSV.
        """
        filepath = self._generate_filename(name, versioned)
        df.to_csv(filepath, index=False)
        print(f"Features guardados en: {filepath}")
        return filepath

    def list_feature_versions(self, name: str = "forex_features") -> list:
        """
        Lista todas las versiones guardadas de un dataset de features.
        """
        files = [
            f for f in os.listdir(self.preprocessed_dir)
            if f.startswith(name) and f.endswith(".csv")
        ]
        files.sort(reverse=True)
        return files

    def load_latest_features(self, name: str = "forex_features") -> pd.DataFrame:
        """
        Carga el CSV más reciente (última versión).
        """
        files = self.list_feature_versions(name)
        if not files:
            raise FileNotFoundError(f"No se encontraron versiones para '{name}' en {self.preprocessed_dir}")
        latest_file = os.path.join(self.preprocessed_dir, files[0])
        print(f"Cargando última versión: {latest_file}")
        return pd.read_csv(latest_file)

    def load_specific_version(self, filename: str) -> pd.DataFrame:
        """
        Carga una versión específica del archivo de features.
        """
        filepath = os.path.join(self.preprocessed_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"El archivo {filename} no existe en {self.preprocessed_dir}")
        print(f"Cargando versión específica: {filepath}")
        return pd.read_csv(filepath)
