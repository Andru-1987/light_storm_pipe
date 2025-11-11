import os
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)



class ModelTester:
    """
    Clase encargada de testear el modelo entrenado:
    - Generar predicciones
    - Calcular métricas
    - Guardarlas en CSV/JSON
    - Guardar imágenes de resultados (matriz de confusión, etc.)
    Todo dentro de una carpeta única por corrida.
    """
    
    def __init__(self, model_path, X_test, y_test, label_names=None, base_dir="artifacts/test_runs"):
        self.model_path = model_path
        self.X_test = X_test
        self.y_test = y_test
        self.label_names = label_names
        self.base_dir = base_dir

        # Crear carpeta específica por corrida
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Configurar logging de la corrida
        logging.basicConfig(
            filename=os.path.join(self.run_dir, "test.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.model = self._load_model()
        logging.info(f"Inicializada prueba en {self.run_dir}")

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
            "F1 Macro": f1_score(self.y_test, y_pred, average="macro"),
        }

        # Guardar métricas y predicciones en disco
        self.save_metrics(metrics)
        self.save_predictions(y_pred)
        self.plot_confusion_matrix(y_pred)

        logging.info(f"Métricas finales: {metrics}")
        return metrics, y_pred

    def save_metrics(self, metrics):
        """
        Guarda las métricas en un CSV y JSON dentro de la carpeta de la corrida.
        """
        metrics_path_csv = os.path.join(self.run_dir, "metrics.csv")
        metrics_path_json = os.path.join(self.run_dir, "metrics.json")

        df = pd.DataFrame([metrics])
        df.to_csv(metrics_path_csv, index=False)
        df.to_json(metrics_path_json, orient="records", indent=4)

        logging.info(f"Métricas guardadas en {self.run_dir}")

    def save_predictions(self, y_pred):
        """
        Guarda las predicciones junto con las etiquetas reales.
        """
        preds_df = pd.DataFrame({
            "y_true": self.y_test,
            "y_pred": y_pred
        })
        preds_path = os.path.join(self.run_dir, "predictions.csv")
        preds_df.to_csv(preds_path, index=False)
        logging.info(f"Predicciones guardadas en {preds_path}")

    def plot_confusion_matrix(self, y_pred):
        """
        Dibuja y guarda la matriz de confusión.
        """
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_names)
        disp.plot(cmap="Blues", values_format="d", colorbar=False)
        plt.title("Matriz de Confusión - Test")
        plt.tight_layout()

        fig_path = os.path.join(self.run_dir, "confusion_matrix.png")
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Matriz de confusión guardada en {fig_path}")
