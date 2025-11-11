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
            print(f"\nModelos y scalers guardados correctamente en:\n{MODEL_DIR}")

        except Exception as e:
            logging.error(f"Error al guardar modelos o scalers: {e}")
            print(f"Error al guardar modelos o scalers: {e}")
