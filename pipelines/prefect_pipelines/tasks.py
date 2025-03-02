import io
from typing import Literal

from tqdm import tqdm
import mlflow
import boto3
from prefect import task
import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit


@task
def find_merchant_folders(input_folder):
    return [os.path.join(input_folder, d) for d in os.listdir(input_folder)
            if os.path.isdir(os.path.join(input_folder, d)) and d.startswith("merchant")]


@task
def get_orders_by_merchant(folder_path):
    orders_folder = os.path.join(folder_path, "orders")

    usecols = ["transaction_id", "created_at", "merchant_id"]
    dtypes = {"transaction_id": "object", "merchant_id": "object"}
    orders_df = pd.concat([pd.read_csv(os.path.join(orders_folder, f),
                                       usecols=usecols,
                                       parse_dates=["created_at"],
                                       dtype=dtypes) for f in tqdm(os.listdir(orders_folder)) if f.endswith(".csv")],
                           ignore_index=True) if os.path.exists(orders_folder) else pd.DataFrame()
    return orders_df


@task
def get_orders_df(input_folder):
    merchant_folders = find_merchant_folders(input_folder)
    dataframes = [get_orders_by_merchant(folder) for folder in merchant_folders]

    # Stack all merchant DataFrames together
    final_df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    return final_df


@task
def get_chargebacks_df(input_folder):
    merchant_folders = find_merchant_folders(input_folder)

    usecols = ["transaction_id", "created_at"]
    dtypes = {"transaction_id": "object"}

    dfs = []
    for folder in merchant_folders:
        merchant_id = os.path.basename(folder).split("_")[1]
        merchant_chargebacks = pd.read_csv(os.path.join(folder, "chargebacks.csv"),
                               usecols=usecols,
                               parse_dates=["created_at"],
                               dtype=dtypes)
        merchant_chargebacks["merchant_id"] = merchant_id
        dfs.append(merchant_chargebacks)
    return pd.concat(dfs, ignore_index=True)


@task
def get_fraud_alerts_df(input_folder):
    merchant_folders = find_merchant_folders(input_folder)

    usecols = ["transaction_id", "created_at"]
    dtypes = {"transaction_id": "object"}

    dfs = []
    for folder in merchant_folders:
        merchant_id = os.path.basename(folder).split("_")[1]
        merchant_fraud_alerts = pd.read_csv(os.path.join(folder, "fraud_alerts.csv"),
                               usecols=usecols,
                               parse_dates=["created_at"],
                               dtype=dtypes)
        merchant_fraud_alerts["merchant_id"] = merchant_id
        dfs.append(merchant_fraud_alerts)
    return pd.concat(dfs, ignore_index=True)


@task
def aggregate_and_merge(orders_df: pd.DataFrame,
                        chargebacks_df: pd.DataFrame,
                        fraud_alerts_df: pd.DataFrame) -> pd.DataFrame:
    orders_df["created_at"] = pd.to_datetime(orders_df["created_at"]).dt.floor("D")
    chargebacks_df["created_at"] = pd.to_datetime(chargebacks_df["created_at"]).dt.floor("D")
    fraud_alerts_df["created_at"] = pd.to_datetime(fraud_alerts_df["created_at"]).dt.floor("D")

    orders_agg = orders_df.groupby(["merchant_id", "created_at"]).agg(orders_count=("transaction_id", "count")).reset_index()
    orders_agg.rename(columns={"created_at": "date"}, inplace=True)

    chargebacks_agg = chargebacks_df.groupby(["merchant_id", "created_at"]).agg(chargebacks_count=("transaction_id", "count")).reset_index()
    chargebacks_agg.rename(columns={"created_at": "date"}, inplace=True)

    fraud_alerts_agg = fraud_alerts_df.groupby(["merchant_id", "created_at"]).agg(fraud_alerts_count=("transaction_id", "count")).reset_index()
    fraud_alerts_agg.rename(columns={"created_at": "date"}, inplace=True)

    df = pd.merge(orders_agg, chargebacks_agg, on=["merchant_id", "date"], how="outer")
    df = pd.merge(df, fraud_alerts_agg, on=["merchant_id", "date"], how="outer")
    df[["orders_count", "chargebacks_count", "fraud_alerts_count"]] = df[["orders_count", "chargebacks_count", "fraud_alerts_count"]].fillna(0)
    df["date"] = pd.to_datetime(df["date"])

    return df


@task
def compute_rolling_risk(group, window=30):
    group = group.sort_values("date")
    group["orders_roll_sum_30"] = group["orders_count"].rolling(window=window, min_periods=1).sum().shift(1)
    group["risk_events_roll_sum_30"] = (group["chargebacks_count"] + group["fraud_alerts_count"]).rolling(window=window, min_periods=1).sum().shift(1)
    group["rolling_risk_metric_30"] = 100 * (group["risk_events_roll_sum_30"] / (group["orders_roll_sum_30"] + 1e-6))
    return group


@task
def create_extended_lag_features(group, lags=[1, 3, 5, 7, 14, 30, 60, 90]):
    group = group.sort_values("date")
    # Create lags for orders, chargebacks, fraud alerts and their sum (risk events)
    for lag in lags:
        group[f'orders_lag_{lag}'] = group['orders_count'].shift(lag)
        group[f'chargebacks_lag_{lag}'] = group['chargebacks_count'].shift(lag)
        group[f'fraud_alerts_lag_{lag}'] = group['fraud_alerts_count'].shift(lag)
        group[f'risk_events_lag_{lag}'] = (group["chargebacks_count"] + group["fraud_alerts_count"]).shift(lag)
    return group


@task
def create_multiple_window_features(group, windows=[7, 14, 30, 60, 90]):
    group = group.sort_values("date")
    for w in windows:
        group[f'orders_roll_sum_{w}'] = group['orders_count'].rolling(window=w, min_periods=1).sum().shift(1)
        group[f'risk_events_roll_sum_{w}'] = (group['chargebacks_count'] + group['fraud_alerts_count']).rolling(window=w, min_periods=1).sum().shift(1)
    return group


@task
def create_exponential_decay_features(group, span=7):
    group = group.sort_values("date")
    group["orders_ewm_mean"] = group["orders_count"].ewm(span=span, adjust=False).mean().shift(1)
    group["chargebacks_ewm_mean"] = group["chargebacks_count"].ewm(span=span, adjust=False).mean().shift(1)
    group["fraud_alerts_ewm_mean"] = group["fraud_alerts_count"].ewm(span=span, adjust=False).mean().shift(1)
    group["risk_events_ewm_mean"] = (group["chargebacks_count"] + group["fraud_alerts_count"]).ewm(span=span, adjust=False).mean().shift(1)
    group["rolling_risk_metric_ewm"] = 100 * (group["risk_events_ewm_mean"] / (group["orders_ewm_mean"] + 1e-6))
    return group


@task
def create_ratio_features(group, windows=[7, 14, 30, 60, 90]):
    group = group.sort_values("date")
    for w in windows:
        group[f'risk_to_orders_ratio_{w}'] = group[f'risk_events_roll_sum_{w}'] / (group[f'orders_roll_sum_{w}'] + 1e-6)
    # Example differences between consecutive windows
    if "risk_to_orders_ratio_14" in group.columns and "risk_to_orders_ratio_7" in group.columns:
        group['risk_ratio_diff_14_7'] = group['risk_to_orders_ratio_14'] - group['risk_to_orders_ratio_7']
    if "risk_to_orders_ratio_30" in group.columns and "risk_to_orders_ratio_14" in group.columns:
        group['risk_ratio_diff_30_14'] = group['risk_to_orders_ratio_30'] - group['risk_to_orders_ratio_14']
    return group


@task
def create_time_based_and_seasonal_features(df: pd.DataFrame):
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["date_ordinal"] = df["date"].map(pd.Timestamp.toordinal)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


@task
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    transforms = [compute_rolling_risk,
                  create_extended_lag_features,
                  create_multiple_window_features,
                  create_exponential_decay_features,
                  create_ratio_features]
    for transform in transforms:
        df = df.groupby("merchant_id", group_keys=False).apply(transform)

    df = create_time_based_and_seasonal_features(df)

    for lag in [1, 2, 3]:
        df[f'rolling_risk_metric_lag_{lag}'] = df.groupby("merchant_id")["rolling_risk_metric_30"].shift(lag)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


@task
def save_dataframe_to_s3(df: pd.DataFrame, bucket_name: str, file_key: str, save_type: Literal["csv", "parquet"]):
    buffer = io.BytesIO()
    if save_type == "parquet":
        df.to_parquet(buffer, engine='pyarrow', index=False)
    elif save_type == "csv":
        df.to_csv(buffer, index=False)
    buffer.seek(0)

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    s3_client.upload_fileobj(buffer, bucket_name, file_key)
    print(f"File saved to s3://{bucket_name}/{file_key}")


@task
def read_dataframe_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:
    s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

    # Download file into a buffer
    buffer = io.BytesIO()
    s3_client.download_fileobj(bucket_name, file_key, buffer)
    buffer.seek(0)  # Reset buffer position

    # Read Parquet file into DataFrame
    if file_key.endswith(".parquet"):
        df = pd.read_parquet(buffer, engine='pyarrow')
    elif file_key.endswith(".csv"):
        df = pd.read_csv(buffer)
    return df


@task
def get_training_data(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    return df[feature_cols], df[target_col]


@task
def select_hyperparameters(data, target):
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        n_estimators = trial.suggest_int("n_estimators", 100, 800, step=100)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)
        reg_lambda = trial.suggest_float("reg_lambda", 1.0, 100.0)

        model = XGBRegressor(
            random_state=42,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=-1
        )

        fold_scores = []
        for train_idx, val_idx in tscv.split(data):
            X_train, X_val = data.iloc[train_idx], data.iloc[val_idx]
            y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = r2_score(y_val, preds)
            fold_scores.append(score)
        return np.mean(fold_scores)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_trial.params
    print("Best hyperparameters:", best_params)
    print("Best CV R²:", study.best_trial.value)
    return best_params


@task
def train_model(data: pd.DataFrame, target: pd.Series, best_params: dict):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("int20h-xdboobs")
    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME")):
        final_model = XGBRegressor(random_state=42, **best_params)
        final_model.fit(data, target)

        train_r2 = r2_score(target, final_model.predict(data))
        mlflow.log_metric("R2", train_r2)
        mlflow.xgboost.log_model(final_model, "xgboost_model")


@task
def get_feature_columns():
    feature_columns = [
    "rolling_risk_metric_lag_1", "rolling_risk_metric_lag_2", "rolling_risk_metric_lag_3",
    # Extended lag features
    "orders_lag_1", "orders_lag_3", "orders_lag_5", "orders_lag_7", "orders_lag_14", "orders_lag_30",
    "chargebacks_lag_1", "chargebacks_lag_3", "chargebacks_lag_5", "chargebacks_lag_7", "chargebacks_lag_14", "chargebacks_lag_30",
    "fraud_alerts_lag_1", "fraud_alerts_lag_3", "fraud_alerts_lag_5", "fraud_alerts_lag_7", "fraud_alerts_lag_14", "fraud_alerts_lag_30",
    "risk_events_lag_1", "risk_events_lag_3", "risk_events_lag_5", "risk_events_lag_7", "risk_events_lag_14", "risk_events_lag_30",
    # Rolling sums over multiple windows
    "orders_roll_sum_7", "orders_roll_sum_14", "orders_roll_sum_30", "orders_roll_sum_60", "orders_roll_sum_90",
    "risk_events_roll_sum_7", "risk_events_roll_sum_14", "risk_events_roll_sum_30", "risk_events_roll_sum_60", "risk_events_roll_sum_90",
    # Ratio features
    "risk_to_orders_ratio_7", "risk_to_orders_ratio_14", "risk_to_orders_ratio_30", "risk_to_orders_ratio_60", "risk_to_orders_ratio_90",
    "risk_ratio_diff_14_7", "risk_ratio_diff_30_14",
    # Exponential decay features
    "orders_ewm_mean", "chargebacks_ewm_mean", "fraud_alerts_ewm_mean", "risk_events_ewm_mean", "rolling_risk_metric_ewm",
    # Time and seasonal features
    "day_of_week", "month", "date_ordinal",
    "sin_day", "cos_day", "sin_month", "cos_month"
    ]
    return feature_columns


@task
def create_test_df(submission_df: pd.DataFrame, train_df: pd.DataFrame):
    submission_df["merchant_id"] = submission_df["merchant_id_day"].apply(lambda x: x.split("_")[0])
    submission_df["date"] = pd.to_datetime(submission_df["merchant_id_day"].apply(lambda x: x.split("_")[1]))

    feature_columns = get_feature_columns()

    test_rows = []
    for idx, row in submission_df.iterrows():
        merchant_id = row["merchant_id"]
        forecast_date = row["date"]

        merchant_hist = train_df[train_df["merchant_id"] == merchant_id]
        if merchant_hist.empty:
            baseline = {feat: 0 for feat in feature_columns}
        else:
            baseline = merchant_hist.sort_values("date").iloc[-1].to_dict()

        baseline["date_ordinal"] = forecast_date.toordinal()
        baseline["day_of_week"] = forecast_date.dayofweek
        baseline["month"] = forecast_date.month

        test_rows.append(baseline)

    return pd.DataFrame(test_rows)[feature_columns]


