import os
import logging
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from model.preprocessor import Preprocessor
from modules.model.tester import ModelTester
from modules.model.trainer import ModelTrainer


class PipelineRunner:
    """
    Clase orquestadora que ejecuta el flujo completo de entrenamiento y evaluación del modelo.
    
    1. Preprocesamiento de datos
    2. Entrenamiento del modelo
    3. Guardado de artifacts (modelo y scaler)
    4. Evaluación y generación de métricas
    """

    def __init__(self, df: pd.DataFrame, model_class=None, target_col="target_encoded"):
        self.df = df
        self.target_col = target_col
        self.model_class = model_class
        self.model_dir = "artifacts/model"
        self.metrics_dir = "artifacts/test_runs"

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(self.model_dir, "pipeline.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def run(self):
        """
        Ejecuta el flujo completo y devuelve las métricas finales.
        """
        try:
            logging.info(" Inicio del pipeline completo ")

            # 1. Preprocesamiento
            logging.info("Preprocesando datos...")
            pre = Preprocessor(self.df, target_col=self.target_col)
            X_train, X_test, y_train, y_test = pre.split_data()
            X_train_scaled, X_test_scaled = pre.scale(X_train, X_test)

            # 2. Entrenamiento
            logging.info("Entrenando modelo...")
            trainer = ModelTrainer(model=self.model_class)
            model = trainer.train(X_train_scaled, y_train)

            # 3. Guardado de artifacts
            trainer.save_artifacts(model, pre.scaler)

            # 4. Evaluación
            logging.info("Evaluando modelo...")
            tester = ModelTester(
                model_path=os.path.join(self.model_dir, "model_eurgbp_logistic.pkl"),
                X_test=X_test_scaled,
                y_test=y_test,
                label_names=["Down", "Uncertain", "Up"]
            )

            metrics, _ = tester.run_test()

            logging.info(" Pipeline completado exitosamente ")
            return metrics

        except Exception as e:
            logging.error(f"Error en el pipeline: {e}")
            raise
