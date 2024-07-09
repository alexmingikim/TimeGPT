import pandas as pd
import numpy as np
from torchmetrics.functional import symmetric_mean_absolute_percentage_error
import torch
import re
from dotenv import load_dotenv
load_dotenv()
from nixtla import NixtlaClient
nixtla_client = NixtlaClient()
from sys import *
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

def sMAPE(forecast, real):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (sMAPE) between actual and predicted values.
    """
    smape = symmetric_mean_absolute_percentage_error(
        torch.Tensor(list(forecast)), torch.Tensor(list(real))
    )
    return np.float64(smape)

def main():
    # Get arguments
    state = argv[1]
    split_week = datetime.strptime(argv[2], "%Y-%m-%d").date()

    # Use a regular expression to add a space before any uppercase letter that is not at the start
    state = re.sub(r'(?<!^)(?=[A-Z])', ' ', state)

    # Import data 
    df = pd.read_csv("./data/all-states-2010-2024.csv", skiprows=1)

    # Import school holidays
    school_holidays = pd.read_csv("./data/school_holidays.csv")

    # Convert Split_week column to datetime
    school_holidays['Split_week'] = pd.to_datetime(school_holidays['Split_week'], dayfirst=True)

    ### Preprocessing

    # Create Split_week column
    df["Split_week"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["WEEK"].astype(str) + "-1", format="%G%V-%u"
    ).dt.normalize()

    # Get only rows for state of interest
    filtered_df = df[df["REGION"] == state]

    # Select relevant columns
    filtered_df = filtered_df[["Split_week", "%UNWEIGHTED ILI"]]

    # Make "%UNWEIGHTED ILI" column numeric
    filtered_df["%UNWEIGHTED ILI"] = pd.to_numeric(filtered_df["%UNWEIGHTED ILI"])

    # Divide all values in "%UNWEIGHTED ILI" column by 100
    filtered_df["%UNWEIGHTED ILI"] = filtered_df["%UNWEIGHTED ILI"] / 100

    # Drop duplicates
    filtered_df = filtered_df.drop_duplicates(subset=["Split_week"], keep="first")

    # Reset index
    filtered_df = filtered_df.reset_index(drop=True)

    # Convert split_week to datetime
    split_week_dt = pd.to_datetime(split_week)

    # Use split week to create training set
    train_data = filtered_df[filtered_df["Split_week"] < split_week_dt]

    # Create test set
    test_data = filtered_df[filtered_df["Split_week"] >= split_week_dt]

    # Create future dataframe with exogenous features
    future_holidays = school_holidays[school_holidays["Split_week"] >= split_week_dt].head(52)

    # Add exogenous features to training set
    past_holidays = school_holidays[school_holidays["Split_week"] < split_week_dt]
    train_data = train_data.merge(past_holidays)

    ### Forecasting
    quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    timegpt_fcst_df = nixtla_client.forecast(
        df=train_data, 
        quantiles=quantiles,
        h=52, # forecast for 52 weeks ahead
        freq='W-MON', 
        time_col='Split_week', 
        target_col='%UNWEIGHTED ILI',
        X_df=future_holidays
    )

    # Save forecasts and evaluation metrics 
    horizons = [1,4,13,26,52]

    for horizon in horizons:
        ### Save forecasts
        # Define path
        output_dir = f"output/%UNWEIGHTED ILI/zeroshot_covariates/{state}/{horizon}weeks/forecasts"
        output_file = f"{output_dir}/{split_week}.csv"

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract first n weeks of forecasts
        forecasts = timegpt_fcst_df.head(horizon)

        # Save forecasts as csv 
        forecasts.to_csv(output_file, index=False)

        ### Evaluation
        # Merge forecasts with test set for evaluation
        evaluation = test_data.head(horizon).copy()
        evaluation.loc[:, 'TimeGPT'] = forecasts['TimeGPT'].values

        # Calculate evaluation metrics
        mae = mean_absolute_error(evaluation['%UNWEIGHTED ILI'], evaluation['TimeGPT'])
        mape = mean_absolute_percentage_error(evaluation['%UNWEIGHTED ILI'], evaluation['TimeGPT'])
        smape = sMAPE(evaluation['%UNWEIGHTED ILI'], evaluation['TimeGPT'])
        rmse = np.sqrt(mean_squared_error(evaluation['%UNWEIGHTED ILI'], evaluation['TimeGPT']))
        mase = mean_absolute_scaled_error(
            y_true=evaluation['%UNWEIGHTED ILI'],
            y_pred=evaluation['TimeGPT'],
            y_train=train_data['%UNWEIGHTED ILI']
        )

        # Define evaluation metrics - MAE, MAPE, sMAPE, RMSE, MASE
        evaluation_metrics = {
            'Split_week': split_week,
            'MAE': mae,
            'MAPE': mape,
            'sMAPE': smape,
            'RMSE': rmse,
            'MASE': mase
        }

        ### Save evaluation metrics
        # Define path
        eval_dir = f"output/%UNWEIGHTED ILI/zeroshot_covariates/{state}/{horizon}weeks/evaluation"
        eval_file = f"{eval_dir}/{split_week}.csv"

        # Create directory if it doesn't exist
        os.makedirs(eval_dir, exist_ok=True)

        # Create new DataFrame with evaluation metrics
        eval_df = pd.DataFrame(evaluation_metrics, index=[0])

        # Save evaluation metrics as csv
        eval_df.to_csv(eval_file, index=False)

if __name__ == "__main__":
    main()


