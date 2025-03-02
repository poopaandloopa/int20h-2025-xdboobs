from .tasks import (get_orders_df,
                    get_chargebacks_df,
                    get_fraud_alerts_df,
                    aggregate_and_merge,
                    create_features,
                    save_dataframe_to_s3,
                    read_dataframe_from_s3,
                    get_feature_columns,
                    get_training_data,
                    select_hyperparameters,
                    train_model)
import os
from prefect import flow
import pandas as pd


@flow(name="data-processing-pipeline", log_prints=True)
def data_processing_pipeline(data_folder: str) -> pd.DataFrame:
    print("Prepare orders df")
    orders_df = get_orders_df(data_folder)
    print("Prepare chargebacks df")
    chargebacks_df = get_chargebacks_df(data_folder)
    print("Prepare fraud alerts df")
    fraud_alerts_df = get_fraud_alerts_df(data_folder)

    print("Aggregate and merge")
    df = aggregate_and_merge(orders_df, chargebacks_df, fraud_alerts_df)
    print("Create new features")
    df = create_features(df)
    print("Saving to S3")
    save_dataframe_to_s3(df, os.getenv("BUCKET_NAME"), "data/processed_data.parquet", save_type="parquet")

    return df


@flow(name="train-model-pipeline", log_prints=True)
def train_model_pipeline():
    processed_df = read_dataframe_from_s3(os.getenv("BUCKET_NAME"), "data/processed_data.parquet")

    feature_columns = get_feature_columns()
    target_column = "rolling_risk_metric_30"

    data, target = get_training_data(processed_df, feature_columns, target_column)
    best_params = select_hyperparameters(data, target)
    train_model(data, target, best_params)


if __name__ == "__main__":
    data_processing_pipeline("../data")
    train_model_pipeline()
