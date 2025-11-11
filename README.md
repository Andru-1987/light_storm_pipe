```bash
│
├── data/                          # Datos crudos y procesados
│   ├── raw/                       # Datos descargados directamente desde la API
│   └── processed/                 # Datos limpios y con features generadas
│
├── notebooks/                     # Jupyter notebooks (EDA y modelado)
│   ├── 01_data_exploration.ipynb  # Exploración de datos y visualización
│   ├── 02_feature_engineering.ipynb # Creación y análisis de features
│   └── 03_model_training.ipynb    # Entrenamiento y evaluación del modelo
│
├── src/                           # Código fuente modular
│   ├── __init__.py
│   ├── config.py                  # Configuración general (paths, parámetros)
│   ├── data/                      # Módulo de extracción y preparación de datos
│   │   ├── __init__.py
│   │   ├── fetch_data.py          # Llamadas a la API de Alpha Vantage
│   │   ├── preprocess.py          # Limpieza y transformaciones básicas
│   │   └── features.py            # Cálculo de indicadores técnicos (RSI, SMA, ATR, etc.)
│   │
│   ├── model/                     # Módulo de entrenamiento e inferencia
│   │   ├── __init__.py
│   │   ├── train.py               # Entrenamiento del modelo
│   │   ├── evaluate.py            # Evaluación y métricas
│   │   └── predict.py             # Predicción y carga de modelos entrenados
│   │
│   ├── cli/                       # Aplicación de línea de comandos (CLI)
│   │   ├── __init__.py
│   │   ├── main.py                # Punto de entrada del comando principal
│   │   ├── scheduler.py           # Lógica de ejecución diaria (CRON, UTC check)
│   │   └── logger_config.py       # Configuración central del logging
│   │
│   └── utils/                     # Utilidades generales
│       ├── __init__.py
│       ├── time_utils.py          # Funciones auxiliares para manejo de fechas
│       ├── file_utils.py          # Lectura/escritura de CSVs y paths
│       └── indicators.py          # Implementaciones personalizadas de indicadores
│
├── models/                        # Modelos entrenados
│   ├── model_eurgbp.pkl
│   └── model_usdjpy.pkl
│
├── logs/                          # Archivos de logging
│   └── app.log
│
├── outputs/                       # Resultados de predicciones diarias
│   └── predictions.csv
│
├── requirements.txt               # Dependencias del proyecto
├── Dockerfile                     # Imagen base del proyecto
├── docker-compose.yaml            # (opcional) Orquestación si hay más servicios
├── README.md                      # Documentación de instalación y uso
└── .env.example                   # Variables de entorno (API key, rutas, etc.)

```

Como deberia correr? 

```bash
docker compose run ml_service --train-model
```

```bash
docker compose run ml_service --inference
```
