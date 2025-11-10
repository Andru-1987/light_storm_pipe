import os
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Directorio donde se guardan las métricas
METRICS_DIR = "E:/data_lightstorm/metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

# Configuración básica de logging
logging.basicConfig(
    filename=os.path.join(METRICS_DIR, "testing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelTester:
    """
    Clase encargada de testear el modelo entrenado:
    - Generar predicciones
    - Calcular métricas
    - Guardarlas en CSV
    - (Opcional) Mostrar matriz de confusión
    """
    def __init__(self, model_path, X_test, y_test, label_names=None):
        self.model_path = model_path
        self.X_test = X_test
        self.y_test = y_test
        self.label_names = label_names
        self.model = self._load_model()

    def _load_model(self):
        logging.info(f"Cargando modelo desde {self.model_path}")
        model = joblib.load(self.model_path)
        return model

    def run_test(self):
        """
        Ejecuta las predicciones y calcula las métricas principales.
        """
        logging.info("Iniciando test del modelo...")
        y_pred = self.model.predict(self.X_test)

        metrics = {
            "Model": os.path.basename(self.model_path),
            "Balanced Accuracy": balanced_accuracy_score(self.y_test, y_pred),
            "F1 Macro": f1_score(self.y_test, y_pred, average="macro")
        }

        logging.info(f"Métricas obtenidas: {metrics}")
        return metrics, y_pred

    def save_metrics(self, metrics, filename="test_results.csv"):
        """
        Guarda las métricas en un CSV dentro de la carpeta /metrics.
        """
        path = os.path.join(METRICS_DIR, filename)
        df = pd.DataFrame([metrics])

        if os.path.exists(path):
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            df.to_csv(path, index=False)

        logging.info(f"Métricas guardadas en {path}")
        print(f"✅ Métricas guardadas en: {path}")

    def plot_confusion_matrix(self, y_pred):
        """
        Dibuja la matriz de confusión.
        """
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_names)
        disp.plot(cmap="Blues", values_format="d", colorbar=False)
        plt.title("Matriz de Confusión - Test")
        plt.tight_layout()
        plt.show()
