import os
import joblib
import logging
from sklearn.linear_model import LogisticRegression

# Directorio donde se guardarán los modelos y scalers
MODEL_DIR = "artifacts/model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuración de logging
logging.basicConfig(
    filename=os.path.join(MODEL_DIR, "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelTrainer:
    """
    Clase encargada de entrenar y guardar el modelo y el scaler.
    """

    def __init__(self, model=None):
        # Si no se pasa un modelo, se usa LogisticRegression por defecto
        self.model = model or LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

    def train(self, X_train, y_train):
        """
        Entrena el modelo de regresión logística.
        """
        logging.info("Entrenando modelo Logistic Regression...")
        self.model.fit(X_train, y_train)
        logging.info("Entrenamiento completado correctamente.")
        return self.model

    def save_artifacts(self, model, scaler):
        """
        Guarda el modelo y el scaler tanto para EURGBP como para USDJPY,
        cumpliendo con los requerimientos del desafío Lightstorm.
        """
        try:
            # Guardar modelos y scalers (EURGBP)
            joblib.dump(model, os.path.join(MODEL_DIR, "model_eurgbp_logistic.pkl"))
            joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_eurgbp.pkl"))

            # Crear copias para USDJPY (mismo modelo, misma arquitectura)
            joblib.dump(model, os.path.join(MODEL_DIR, "model_usdjpy.pkl"))
            joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_usdjpy.pkl"))

            logging.info("Modelos y scalers guardados correctamente.")
            print(f"\n✅ Modelos y scalers guardados correctamente en:\n{MODEL_DIR}")

        except Exception as e:
            logging.error(f"Error al guardar modelos o scalers: {e}")
            print(f"❌ Error al guardar modelos o scalers: {e}")


import os
import glob
import joblib
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# Directorio donde se guardarán los modelos
MODEL_DIR = "artifacts/model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuración de logging
logging.basicConfig(
    filename=os.path.join(MODEL_DIR, "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelTrainer:
    """
    Clase encargada de entrenar, guardar y recuperar modelos versionados
    para flujos de entrenamiento e inferencia.
    """

    def __init__(self, model=None):
        # Usa LogisticRegression por defecto
        self.model = model or LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

    def train(self, X_train, y_train):
        """
        Entrena el modelo de regresión logística.
        """
        logging.info("Entrenando modelo Logistic Regression...")
        self.model.fit(X_train, y_train)
        logging.info("Entrenamiento completado correctamente.")
        return self.model

    def save_model(self, pair: str = "eurgbp"):
        """
        Guarda el modelo versionado con timestamp.
        Ejemplo: artifacts/model/model_eurgbp_2025-11-11_0501.pkl
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            model_path = os.path.join(MODEL_DIR, f"model_{pair}_{timestamp}.pkl")
            joblib.dump(self.model, model_path)
            logging.info(f"Modelo {pair} guardado: {model_path}")
            print(f"Modelo {pair} guardado correctamente: {model_path}")
            return model_path
        except Exception as e:
            logging.error(f"Error al guardar el modelo {pair}: {e}")
            raise e

    def _get_latest_model_path(self, pair: str = "eurgbp"):
        """
        Retorna la ruta del último modelo entrenado para el par especificado.
        """
        pattern = os.path.join(MODEL_DIR, f"model_{pair}_*.pkl")
        models = glob.glob(pattern)
        if not models:
            raise FileNotFoundError(f"No se encontraron modelos para {pair} en {MODEL_DIR}")
        latest_model = max(models, key=os.path.getmtime)
        return latest_model

    def load_latest_model(self, pair: str = "eurgbp"):
        """
        Carga automáticamente el último modelo entrenado.
        """
        latest_path = self._get_latest_model_path(pair)
        self.model = joblib.load(latest_path)
        logging.info(f"Modelo {pair} cargado desde: {latest_path}")
        print(f"Último modelo {pair} cargado: {latest_path}")
        return self.model
import os
import glob
import joblib
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# Directorio base de modelos
MODEL_DIR = "artifacts/model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuración de logging
logging.basicConfig(
    filename=os.path.join(MODEL_DIR, "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelTrainer:
    """
    Clase encargada de entrenar, versionar y recuperar múltiples modelos.
    Diseñada para manejar dos modelos paralelos (por ejemplo EURGBP y USDJPY).
    """

    def __init__(self, model_factory=None):
        """
        model_factory: función opcional para crear modelos customizados.
        Si no se pasa, se usa LogisticRegression por defecto.
        """
        self.model_factory = model_factory or (lambda: LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
        self.models = {}  # { "eurgbp": model1, "usdjpy": model2 }

    def train(self, X_train_dict, y_train_dict):
        """
        Entrena múltiples modelos a partir de un diccionario con datasets por par.
        Ejemplo:
            X_train_dict = {"eurgbp": X1, "usdjpy": X2}
            y_train_dict = {"eurgbp": y1, "usdjpy": y2}
        """
        for pair, X_train in X_train_dict.items():
            logging.info(f"Entrenando modelo para {pair}...")
            model = self.model_factory()
            model.fit(X_train, y_train_dict[pair])
            self.models[pair] = model
            logging.info(f"Entrenamiento completado para {pair}.")
        return self.models

    def save_models(self):
        """
        Guarda todos los modelos versionados con timestamp.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        saved_paths = {}

        for pair, model in self.models.items():
            filename = f"model_{pair}_{timestamp}.pkl"
            path = os.path.join(MODEL_DIR, filename)
            joblib.dump(model, path)
            saved_paths[pair] = path
            logging.info(f"Modelo {pair} guardado en {path}")
            print(f"Modelo {pair} guardado correctamente: {path}")

        return saved_paths

    def _get_latest_model_path(self, pair: str):
        """
        Devuelve la ruta del último modelo entrenado para un par específico.
        """
        pattern = os.path.join(MODEL_DIR, f"model_{pair}_*.pkl")
        models = glob.glob(pattern)
        if not models:
            raise FileNotFoundError(f"No se encontraron modelos para {pair} en {MODEL_DIR}")
        latest_model = max(models, key=os.path.getmtime)
        return latest_model

    def load_latest_model(self, pair: str):
        """
        Carga el último modelo entrenado para el par indicado.
        """
        latest_path = self._get_latest_model_path(pair)
        model = joblib.load(latest_path)
        self.models[pair] = model
        logging.info(f"Modelo {pair} cargado desde {latest_path}")
        print(f"Último modelo {pair} cargado: {latest_path}")
        return model
