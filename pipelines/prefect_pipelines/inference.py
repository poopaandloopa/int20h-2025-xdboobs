import os

import pandas as pd
from prefect import flow
from .tasks import create_test_df, read_dataframe_from_s3, save_dataframe_to_s3
import mlflow


@flow(name="inference-pipeline", log_prints=True)
def inference_pipeline(submission_file_s3: str, model_uri: str):
    submission_df = read_dataframe_from_s3(os.getenv("BUCKET_NAME"), submission_file_s3)
    processed_df = read_dataframe_from_s3(os.getenv("BUCKET_NAME"), "data/processed_data.parquet")

    test_df = create_test_df(submission_df, processed_df)
    model = mlflow.pyfunc.load_model(model_uri)

    test_df["risk_metric_pred"] = model.predict(test_df)

    final_submission_df = pd.DataFrame({
        "merchant_id_day": submission_df["merchant_id_day"],
        "risk_metric": test_df["risk_metric_pred"]
    })
    save_dataframe_to_s3(final_submission_df, os.getenv("BUCKET_NAME"), "submission.csv", save_type="csv")


if __name__ == "__main__":
    inference_pipeline(submission_file_s3="sample_submission.csv",
                       model_uri="s3://int20h-data/mlflow_artifacts/531295962442669067/b1c7997fd0bc4a8580f5eb30fffe2027/artifacts/xgboost_model")
