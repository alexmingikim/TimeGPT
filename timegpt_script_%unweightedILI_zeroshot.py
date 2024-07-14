import pandas as pd
import numpy as np
import torch
from torchmetrics.functional import symmetric_mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from datetime import datetime
import os
import re
from dotenv import load_dotenv
from nixtla import NixtlaClient
from sys import argv

load_dotenv()
nixtla_client = NixtlaClient()

def sMAPE(forecast, real):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (sMAPE) between actual and predicted values.
    """
    smape = symmetric_mean_absolute_percentage_error(
        torch.Tensor(list(forecast)), torch.Tensor(list(real))
    )
    return np.float64(smape)

def save_forecasts_and_evaluate(forecasts, real_data, horizon, split_week, state, output_dir_base, train_data):
    # Add real data to forecasts df
    forecasts = forecasts.copy()
    forecasts.loc[:, 'Real'] = real_data['%UNWEIGHTED ILI'].values

    # Reorder columns
    cols = forecasts.columns.tolist()
    cols.insert(1, cols.pop(cols.index('Real')))
    forecasts = forecasts.reindex(columns=cols)

    # Save forecasts as csv
    output_dir = f"{output_dir_base}/{state}/{horizon}week/forecasts"
    output_file = f"{output_dir}/{split_week}.csv"
    os.makedirs(output_dir, exist_ok=True)
    forecasts.to_csv(output_file, index=False)

    # Calculate evaluation metrics
    mae = mean_absolute_error(forecasts['Real'], forecasts['TimeGPT'])
    mape = mean_absolute_percentage_error(forecasts['Real'], forecasts['TimeGPT'])
    smape = sMAPE(forecasts['Real'], forecasts['TimeGPT'])
    rmse = np.sqrt(mean_squared_error(forecasts['Real'], forecasts['TimeGPT']))
    mase = mean_absolute_scaled_error(
        y_true=forecasts['Real'],
        y_pred=forecasts['TimeGPT'],
        y_train=train_data['%UNWEIGHTED ILI']
    )

    evaluation_metrics = {
        'Split_week': split_week,
        'MAE': mae,
        'MAPE': mape,
        'sMAPE': smape,
        'RMSE': rmse,
        'MASE': mase
    }

    # Save evaluation metrics as csv
    eval_dir = f"{output_dir_base}/{state}/{horizon}week/evaluation"
    eval_file = f"{eval_dir}/{split_week}.csv"
    os.makedirs(eval_dir, exist_ok=True)
    eval_df = pd.DataFrame(evaluation_metrics, index=[0])
    eval_df.to_csv(eval_file, index=False)

def main():
    # Get arguments
    state = argv[1]
    split_week = datetime.strptime(argv[2], "%Y-%m-%d").date()

    # Use a regular expression to add a space before any uppercase letter that is not at the start
    state = re.sub(r'(?<!^)(?=[A-Z])', ' ', state)

    # Import data
    df = pd.read_csv("./data/all-states-2010-2024.csv", skiprows=1)

    # Preprocessing
    df["Split_week"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["WEEK"].astype(str) + "-1", format="%G%V-%u"
    ).dt.normalize()
    filtered_df = df[df["REGION"] == state][["Split_week", "%UNWEIGHTED ILI"]]
    filtered_df["%UNWEIGHTED ILI"] = pd.to_numeric(filtered_df["%UNWEIGHTED ILI"]) / 100
    filtered_df = filtered_df.drop_duplicates(subset=["Split_week"], keep="first").reset_index(drop=True)

    # Convert split_week to datetime
    split_week_dt = pd.to_datetime(split_week)

    # Use split week to create training and test sets
    train_data = filtered_df[filtered_df["Split_week"] < split_week_dt]
    test_data = filtered_df[filtered_df["Split_week"] >= split_week_dt]

    # Forecasting
    quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    timegpt_fcst_df = nixtla_client.forecast(
        df=train_data,
        quantiles=quantiles,
        h=52,  # forecast for 52 weeks ahead
        freq='W-MON',
        time_col='Split_week',
        target_col='%UNWEIGHTED ILI'
    )

    # Prediction horizons
    horizons = [1, 4, 13, 26, 52]
    output_dir_base = "output/%UNWEIGHTED ILI/zeroshot/all"

    for horizon in horizons:
        # Extract first n weeks of forecasts
        forecasts = timegpt_fcst_df.head(horizon)
        real_data = test_data.head(horizon).copy()
        save_forecasts_and_evaluate(forecasts, real_data, horizon, split_week, state, output_dir_base, train_data)

    # Winter analysis
    winter_months = [11, 12, 1, 2, 3, 4]
    output_dir_base = "output/%UNWEIGHTED ILI/zeroshot/winter"

    for horizon in [1, 4]: #
        forecasts = timegpt_fcst_df.head(horizon).copy()
        real_data = test_data.head(horizon).copy()
        forecasts['Split_week'] = pd.to_datetime(forecasts['Split_week'])
        forecasts['Month'] = forecasts['Split_week'].dt.month
        filtered_forecasts = forecasts[forecasts['Month'].isin(winter_months)].drop(columns=['Month'])

        if not filtered_forecasts.empty:
            save_forecasts_and_evaluate(filtered_forecasts, real_data, horizon, split_week, state, output_dir_base, train_data)

if __name__ == "__main__":
    main()
