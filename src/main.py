from modules.data.fetch_data import FetchData
from modules.data.pre_processing import ForexFeatureEngineer
from modules.data.upload_feature_store import FeatureStoreManager
from modules.utils.args_helper import ArgsHelper

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

    PipelineRunner(df_features)




def main(params):

    # Obtencion de datos
    
    print(params)


if __name__ == "__main__":

    args_helper = ArgsHelper()
    params = args_helper.get_params()

    args_helper.show_parameters()

    main(params)
