import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import ta
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

class ForexFeatureEngineer:
  
    def __init__(self):
        self.feature_columns = []
        
    def create_technical_features(self, df):
        """
        Crear features técnicas usando la librería ta con Pandas
        """
        print("Creando features técnicas...")
        
        # Asegurar que tenemos las columnas necesarias básicas
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en el DataFrame")
        
        # Verificar si volume existe, si no crear uno sintético
        if 'volume' not in df.columns:
            print("⚠️  Columna 'volume' no encontrada. Creando volume sintético...")
            # Crear volume sintético basado en la volatilidad del día
            df['volume'] = ((df['high'] - df['low']) / df['close']) * 1000000
        
        # Hacer copia para no modificar el original
        df_processed = df.copy()
        
        # 1. INDICADORES DE TENDENCIA (SMA)
        print("  - Calculando SMAs...")
        df_processed['SMA_30'] = ta.trend.sma_indicator(df_processed['close'], window=30)
        df_processed['SMA_90'] = ta.trend.sma_indicator(df_processed['close'], window=90)
        df_processed['SMA_crossover'] = df_processed['SMA_30'] - df_processed['SMA_90']
        df_processed['sma_ratio'] = df_processed['SMA_30'] / df_processed['SMA_90']
        
        # 2. INDICADORES DE MOMENTUM (RSI)
        print("  - Calculando RSI...")
        df_processed['RSI'] = ta.momentum.rsi(df_processed['close'], window=14)
        
        # 3. INDICADORES DE VOLATILIDAD (ATR)
        print("  - Calculando ATR...")
        df_processed['ATR'] = ta.volatility.average_true_range(
            df_processed['high'], df_processed['low'], df_processed['close'], window=14
        )
        
        # 4. VOLATILIDAD HISTÓRICA
        print("  - Calculando volatilidades...")
        df_processed['returns'] = df_processed['close'].pct_change()
        df_processed['volatility_30d'] = df_processed['returns'].rolling(window=30).std()
        df_processed['volatility_rolling'] = df_processed['returns'].rolling(window=10).std()
        
        # 5. FEATURES DE VOLUMEN
        print("  - Calculando features de volumen...")
        df_processed['volume_ratio'] = df_processed['volume'] / df_processed['volume'].rolling(window=30).mean()
        
        # Tendencia de volumen (pendiente de regresión lineal 5 días)
        def calculate_volume_trend(volume_series):
            if len(volume_series) < 5 or volume_series.isna().any():
                return np.nan
            x = np.arange(len(volume_series))
            slope = np.polyfit(x, volume_series, 1)[0]
            return slope
        
        df_processed['volume_trend'] = df_processed['volume'].rolling(window=5).apply(
            calculate_volume_trend, raw=False
        )
        
        # 6. RETORNO DEL DÍA ACTUAL
        df_processed['return_t'] = df_processed['close'].pct_change()
        
        # Limpiar columnas temporales
        df_processed = df_processed.drop(['returns'], axis=1)
        
        return df_processed
    
    def create_temporal_features(self, df, date_column='date'):
        """
        Crear features temporales a partir de la columna de fecha con Pandas
        """
        print("Creando features temporales...")
        
        # Asegurar que la columna de fecha existe
        if date_column not in df.columns:
            raise ValueError(f"Columna de fecha '{date_column}' no encontrada")
        
        # Hacer copia para no modificar el original
        df_processed = df.copy()
        
        # Convertir a datetime si no lo es
        if not pd.api.types.is_datetime64_any_dtype(df_processed[date_column]):
            df_processed[date_column] = pd.to_datetime(df_processed[date_column])
        
        # Extraer componentes temporales
        df_processed['month'] = df_processed[date_column].dt.month
        df_processed['quarter'] = df_processed[date_column].dt.quarter
        df_processed['day_of_week'] = df_processed[date_column].dt.dayofweek  # 0=Lunes, 6=Domingo
        df_processed['is_month_end'] = df_processed[date_column].dt.is_month_end.astype(int)
        
        # Features cíclicas para mes y día de la semana
        df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
        df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
        df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
        df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
        
        return df_processed
    
    def create_target(self, df):
        """
        Crear target de clasificación ternaria con Pandas
        """
        print("Creando variable target...")
        
        # Hacer copia para no modificar el original
        df_processed = df.copy()
        
        # Calcular retorno del día siguiente
        df_processed['return_t1'] = df_processed['close'].shift(-1) / df_processed['close'] - 1
        
        # Definir condiciones para clasificación ternaria
        conditions = [
            df_processed['return_t1'] > 0.001,    # UP: sube más del 0.1%
            df_processed['return_t1'] < -0.001    # DOWN: baja más del 0.1%
        ]
        choices = ['up', 'down']
        
        # Aplicar condiciones
        df_processed['target'] = np.select(conditions, choices, default='neutral')
        
        # Mapear a valores numéricos para el modelo
        target_map = {'down': 0, 'neutral': 1, 'up': 2}
        df_processed['target_encoded'] = df_processed['target'].map(target_map)
        
        return df_processed
    
    def get_feature_columns(self):
        """
        Retorna la lista de columnas de features
        """
        technical_features = [
            'SMA_30', 'SMA_90', 'SMA_crossover', 'RSI', 'ATR', 
            'volatility_30d', 'volatility_rolling', 'sma_ratio', 'return_t',
            'volume_ratio', 'volume_trend'
        ]
        
        temporal_features = [
            'month', 'quarter', 'day_of_week', 'is_month_end',
            'month_sin', 'month_cos', 'day_sin', 'day_cos'
        ]
        
        return technical_features + temporal_features
    
    def prepare_features(self, df, date_column='date'):
        """
        Pipeline completo de feature engineering con Pandas
        """
        print("="*60)
        print("INICIANDO PIPELINE DE FEATURE ENGINEERING")
        print("="*60)
        
        # Hacer copia para no modificar el original
        df_processed = df.copy()
        
        print(f"Dataset de entrada: {df_processed.shape[0]} filas, {df_processed.shape[1]} columnas")
        print(f"Columnas disponibles: {list(df_processed.columns)}")
        
        # Aplicar transformaciones
        df_processed = self.create_technical_features(df_processed)
        df_processed = self.create_temporal_features(df_processed, date_column)
        df_processed = self.create_target(df_processed)
        
        # Obtener lista de columnas de features
        self.feature_columns = self.get_feature_columns()
        
        # Limpiar valores infinitos y NaN
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        
        # Eliminar filas con NaN en features o target
        initial_shape = df_processed.shape[0]
        df_processed = df_processed.dropna(subset=self.feature_columns + ['target_encoded'])
        final_shape = df_processed.shape[0]
        
        print(f"\n📊 RESULTADOS DEL FEATURE ENGINEERING:")
        print(f"   - Filas eliminadas por NaN: {initial_shape - final_shape}")
        print(f"   - Filas finales: {final_shape}")
        print(f"   - Total features creadas: {len(self.feature_columns)}")
        
        # Distribución del target
        target_dist = df_processed['target'].value_counts()
        print("\n🎯 DISTRIBUCIÓN DEL TARGET:")
        for label, count in target_dist.items():
            percentage = (count / len(df_processed)) * 100
            print(f"   - {label}: {count} ({percentage:.1f}%)")
        
        return df_processed