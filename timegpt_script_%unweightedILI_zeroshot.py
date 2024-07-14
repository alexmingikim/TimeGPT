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

    ## Preprocessing

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

    ## Forecasting
    quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    timegpt_fcst_df = nixtla_client.forecast(
        df=train_data, 
        quantiles=quantiles,
        h=52, # forecast for 52 weeks ahead
        freq='W-MON', 
        time_col='Split_week', 
        target_col='%UNWEIGHTED ILI'
    )

    # Prediction horizons
    horizons = [1,4,13,26,52] #

    for horizon in horizons:
        # Define path for storing forecasts
        output_dir = f"output/%UNWEIGHTED ILI/zeroshot/all/{state}/{horizon}week/forecasts" #
        output_file = f"{output_dir}/{split_week}.csv"
        os.makedirs(output_dir, exist_ok=True)

        # Extract first n weeks of forecasts
        forecasts = timegpt_fcst_df.head(horizon)

        # Add real data to forecasts df 
        real_data = test_data.head(horizon).copy()
        forecasts.loc[:, 'Real'] = real_data['%UNWEIGHTED ILI'].values

        # Reorder columns 
        cols = forecasts.columns.tolist()
        cols.insert(1, cols.pop(cols.index('Real')))
        forecasts = forecasts.reindex(columns=cols)

        # Save forecasts as csv 
        forecasts.to_csv(output_file, index=False)

        ## Evaluation
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

        # Define path for storing evaluation metrics
        eval_dir = f"output/%UNWEIGHTED ILI/zeroshot/all/{state}/{horizon}week/evaluation" #
        eval_file = f"{eval_dir}/{split_week}.csv"
        os.makedirs(eval_dir, exist_ok=True)

        # Save evaluation mertrics as csv
        eval_df = pd.DataFrame(evaluation_metrics, index=[0])
        eval_df.to_csv(eval_file, index=False)

    ### Winter analysis 

    # Define winter months (November, December, January, February, March, April)
    winter_months = [11, 12, 1, 2, 3, 4] #

    for horizon in [1,4]: #
        # Define path for storing forecasts
        output_dir = f"output/%UNWEIGHTED ILI/zeroshot/winter/{state}/{horizon}week/forecasts" #
        output_file = f"{output_dir}/{split_week}.csv"
        os.makedirs(output_dir, exist_ok=True)

        # Extract first n weeks of forecasts
        forecasts = timegpt_fcst_df.head(horizon).copy()

        # Add real data to forecasts df 
        real_data = test_data.head(horizon).copy()
        forecasts.loc[:, 'Real'] = real_data['%UNWEIGHTED ILI'].values

        # Reorder columns 
        cols = forecasts.columns.tolist()
        cols.insert(1, cols.pop(cols.index('Real')))
        forecasts = forecasts.reindex(columns=cols)

        # Extract forecasts for winter months 
        forecasts['Split_week'] = pd.to_datetime(forecasts['Split_week'])
        forecasts['Month'] = forecasts['Split_week'].dt.month
        filtered_forecasts = forecasts[forecasts['Month'].isin(winter_months)]
        filtered_forecasts = filtered_forecasts.drop(columns=['Month'])

        # Save forecasts and do evaluation if filtered df is not empty
        if not filtered_forecasts.empty:
            # Save forecasts as csv 
            filtered_forecasts.to_csv(output_file, index=False)

            # Calculate evaluation metrics
            mae = mean_absolute_error(filtered_forecasts['Real'], filtered_forecasts['TimeGPT'])
            mape = mean_absolute_percentage_error(filtered_forecasts['Real'], filtered_forecasts['TimeGPT'])
            smape = sMAPE(filtered_forecasts['Real'], filtered_forecasts['TimeGPT'])
            rmse = np.sqrt(mean_squared_error(filtered_forecasts['Real'], filtered_forecasts['TimeGPT']))
            mase = mean_absolute_scaled_error(
                y_true=filtered_forecasts['Real'],
                y_pred=filtered_forecasts['TimeGPT'],
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

            # Define path for storing evaluation metrics
            eval_dir = f"output/%UNWEIGHTED ILI/zeroshot/winter/{state}/{horizon}week/evaluation" #
            eval_file = f"{eval_dir}/{split_week}.csv"
            os.makedirs(eval_dir, exist_ok=True)

            # Save evaluation metrics as csv
            eval_df = pd.DataFrame(evaluation_metrics, index=[0])
            eval_df.to_csv(eval_file, index=False)

if __name__ == "__main__":
    main()


