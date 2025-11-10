from modules.data.fetch_data import FetchData
from modules.data.pre_processing import ForexFeatureEngineer
from modules.data.upload_feature_store import FeatureStoreManager


def main():

    # Obtencion de datos
    fetcher = FetchData()

    # ver si ya existe el archivo y si ya existe usarlo
    df = fetcher.fetch_raw_data(
        from_symbol="EUR",
        to_symbol="GBP",
        function="FX_DAILY",
        outputsize="full", 
        datatype="json"
    )


    file_path_stored =fetcher.store_raw_data(df, name="eur_gbp_daily")

    # Pre Procesado de datos
    engineer = ForexFeatureEngineer()
    df_features = engineer.prepare_features(df, date_column='timestamp')

    print(df_features.shape)

    store_manager = FeatureStoreManager(".")
    file = store_manager.save_features(df_features)

    print("Guardado en:", file)

if __name__ == "__main__":
    main()
