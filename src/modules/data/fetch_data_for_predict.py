import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging

load_dotenv()


class PredictionDataFetcher:
    """
    Clase enfocada en obtener los datos recientes para la predicción diaria de Forex.
    """

    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_url = os.getenv("API_URL")

        if not self.api_key:
            raise ValueError("No se encontró la variable ALPHAVANTAGE_API_KEY en el archivo .env")

    def _check_time_window(self, start_hour=1, end_hour=4, force=False) -> bool:
        """
        Verifica si la ejecución está dentro de la ventana horaria (UTC).
        Permite ejecución forzada con force=True.
        """
        now = datetime.now(timezone.utc)
        if start_hour <= now.hour < end_hour:
            return True
        elif force:
            logging.warning(f"Ejecución forzada fuera de ventana ({now.hour:02d}:00 UTC).")
            return True
        else:
            logging.warning(f"Fuera de ventana ({now.hour:02d}:00 UTC). Ejecución cancelada.")
            return False

    def fetch_latest_daily_data(self, from_symbol: str, to_symbol: str, force=False) -> pd.DataFrame:
        """
        Obtiene los datos diarios más recientes para un par de divisas.
        Se usa exclusivamente para generar la predicción del cierre T+1.
        """
        if not self._check_time_window(force=force):
            return None

        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "apikey": self.api_key,
            "outputsize": "compact"  # solo los últimos ~100 días
        }

        logging.info(f"Obteniendo datos recientes de {from_symbol}/{to_symbol} desde Alpha Vantage...")
        response = requests.get(self.api_url, params=params)
        if response.status_code != 200:
            logging.error(f"Error HTTP {response.status_code} al llamar a Alpha Vantage")
            return None

        data = response.json()
        if "Time Series FX (Daily)" not in data:
            logging.error("Respuesta inesperada de la API. Falta 'Time Series FX (Daily)'.")
            return None

        df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.rename(columns=lambda c: c.split(". ")[1] if ". " in c else c, inplace=True)

        logging.info(f"Datos obtenidos. Último registro: {df.index[-1].strftime('%Y-%m-%d')}")
        return df
