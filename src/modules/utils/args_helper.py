import argparse
from datetime import datetime


class ArgsHelper:
    """
    Clase auxiliar para gestionar los parámetros del proyecto.

    Parámetros disponibles:
      --train-model     Entrena el modelo (por defecto)
      --inference       Realiza inferencia sobre una fecha específica
      --fecha           Fecha a usar para inferencia (formato YYYY-mm-dd)
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Gestor de argumentos para el proyecto ML",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Definición de argumentos
        mode_group = self.parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--train-model",
            action="store_true",
            help="Ejecuta el entrenamiento del modelo (modo por defecto)"
        )
        mode_group.add_argument(
            "--inference",
            action="store_true",
            help="Ejecuta el modo de inferencia"
        )

        self.parser.add_argument(
            "--fecha",
            type=str,
            default=None,
            help="Fecha a utilizar para inferencia (formato YYYY-mm-dd)"
        )

        # Parseo inicial
        self.args = self.parser.parse_args()
        self._set_defaults()

    def _set_defaults(self):
        # Si no se especifica modo, usar train-model por defecto
        if not self.args.train_model and not self.args.inference:
            self.args.train_model = True

        # Validación simple para fecha
        if self.args.fecha:
            try:
                datetime.strptime(self.args.fecha, "%Y-%m-%d")
            except ValueError:
                self.parser.error("El formato de fecha debe ser YYYY-mm-dd")

    def show_parameters(self):
        """Muestra los parámetros actuales y sus valores."""
        print("Parámetros actuales:")
        for k, v in vars(self.args).items():
            print(f"  {k}: {v}")

    def get_params(self):
        """Devuelve los parámetros como dict (para usar en main)."""
        return vars(self.args)
