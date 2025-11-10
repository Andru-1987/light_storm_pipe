import os
import joblib
import logging
from sklearn.linear_model import LogisticRegression

MODEL_DIR = "E:/data_lightstorm/models"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(MODEL_DIR, "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelTrainer:
    """
    Clase encargada de entrenar y guardar el modelo.
    """
    def __init__(self, model=None):
        self.model = model or LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

    def train(self, X_train, y_train):
        logging.info("Entrenando modelo Logistic Regression...")
        self.model.fit(X_train, y_train)
        logging.info("Entrenamiento completado.")
        return self.model

    def save_model(self, model_name="log_reg.pkl"):
        path = os.path.join(MODEL_DIR, model_name)
        joblib.dump(self.model, path)
        logging.info(f"Modelo guardado en {path}")
        return path
