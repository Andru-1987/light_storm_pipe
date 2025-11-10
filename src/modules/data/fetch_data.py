import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


class FetchData:
    """
    Clase para obtener y almacenar datos crudos desde la API de Alpha Vantage.
    """

    def __init__(self, raw_storage_root: str = "./data/raw/"):
        load_dotenv()
        self.api_url = os.getenv("API_URL")
        self.api_key = os.getenv("API_KEY")

        if not self.api_url:
            raise ValueError("Variable de entorno 'API_URL' no encontrada en .env")
        if not self.api_key:
            raise ValueError("Variable de entorno 'API_KEY' no encontrada en .env")

        self.raw_storage_root = raw_storage_root
        os.makedirs(self.raw_storage_root, exist_ok=True)

    def fetch_raw_data(
        self,
        from_symbol: str = "EUR",
        to_symbol: str = "GBP",
        function: str = "FX_DAILY",
        outputsize: str = "full",
        datatype: str = "json",
    ) -> pd.DataFrame:
        """
        Obtiene datos desde Alpha Vantage según los parámetros indicados.

        Args:
            from_symbol: Símbolo base del par (ej. "EUR")
            to_symbol: Símbolo cotizado del par (ej. "GBP")
            function: Endpoint de Alpha Vantage (por defecto 'FX_DAILY')
            outputsize: 'compact' o 'full'
            datatype: 'json' o 'csv'

        Returns:
            DataFrame con los datos descargados
        """
        params = {
            "function": function,
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "datatype": datatype,
        }

        print(f"Solicitando datos desde {self.api_url} para {from_symbol}/{to_symbol}...")

        response = requests.get(self.api_url, params=params, timeout=30)
        response.raise_for_status()

        # Manejo de respuesta CSV o JSON
        if datatype == "csv":
            df = pd.read_csv(pd.compat.StringIO(response.text))
        else:
            data = response.json()
            if "Time Series FX (Daily)" not in data:
                raise ValueError("La respuesta no contiene 'Time Series FX (Daily)'. "
                                 "Verifique el API key o los parámetros.")
            daily_data = data["Time Series FX (Daily)"]
            df = pd.DataFrame.from_dict(daily_data, orient="index")
            df.reset_index(inplace=True)
            df.rename(columns={"index": "timestamp"}, inplace=True)
            df = df.rename(columns=lambda x: x.split(". ")[-1])  # Limpia nombres como '1. open' -> 'open'
            df = df.astype({
                "open": float,
                "high": float,
                "low": float,
                "close": float
            })
            df["date"] = pd.to_datetime(df["timestamp"])

        print(f"Datos obtenidos: {len(df)} registros.")
        return df

    def store_raw_data(self, df: pd.DataFrame, name: str = "forex_raw") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.csv"
        filepath = os.path.join(self.raw_storage_root, filename)
        df.to_csv(filepath, index=False)
        print(f"Datos almacenados en: {filepath}")
        return filepath
