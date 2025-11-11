from modules.data.fetch_data import FetchData
from modules.data.pre_processing import ForexFeatureEngineer
from modules.data.upload_feature_store import FeatureStoreManager
import argparse



from modules.model.pipe import PipelineRunner

def modeling():
    fetcher = FetchData()

    df = fetcher.fetch_raw_data(
        from_symbol="EUR",
        to_symbol="GBP",
        function="FX_DAILY",
        outputsize="full", 
        datatype="json"
    )


    fetcher.store_raw_data(df, name="eur_gbp_daily")

    # Pre Procesado de datos
    engineer = ForexFeatureEngineer()
    df_features = engineer.prepare_features(df, date_column='timestamp')


    store_manager = FeatureStoreManager(".")
    file=store_manager.save_features(df_features)

    print("Guardado en:", file)

    # FIX tiene 1 valor null que debe ser por el shift --> arreglar 
    df_features.fillna(0, inplace=True)
    training_piper = PipelineRunner(df_features)

    training_piper.run()


def inference():
    # Los pasos que deben pasar en la inferencia: 
    # Tomar los datos desde la ultima fecha desde la api --> PredictionDataFetcher esta clase la hice pero no se si es la correcta
    # preprocesar para dejarlo limpio usando las feaures para preparar la data 

    # pasarlo por el ss guardado 
    # pasarlo por el model para que guarde las predicciones 

    # recorda que las fechas se deben correr en el horario que habian pasado que creo que era entre la 1 y 4 
    pass


def main(args):

    if args.train_model:
        modeling()
        return 

    if args.inference:
        inference()
        return
    

    print("No se ha seleccionado una opcion correcta")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-model", action="store_true", help="Ejecuta el entrenamiento del modelo")
    parser.add_argument("--inference", action="store_true", help="Ejecuta la inferencia")
    args = parser.parse_args()

    main(args)
