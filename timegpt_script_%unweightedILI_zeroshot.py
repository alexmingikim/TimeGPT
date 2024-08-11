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
    Calculate the Symmetric Mean Absolute Percentage Error (sMAPE) between the actual values and the predicted values.
    Parameters:
    - real (list or numpy array): The actual values.
    - forecast (list or numpy array): The predicted values.
    Returns:
    - smape (float): The sMAPE value.
    The sMAPE is a measure of the accuracy of a forecasting model. It is calculated as the average of the absolute percentage differences between the actual and predicted values, with a symmetric weighting.
    """
    smape = symmetric_mean_absolute_percentage_error(
        torch.Tensor([forecast]), torch.Tensor([real])
    )
    return np.float64(smape)

def main():
    # get arguments
    state = argv[1]
    split_week = datetime.strptime(argv[2], "%Y-%m-%d").date()

    # process state name (add a space before any uppercase letter that is not at the start e.g. New York)
    state = re.sub(r'(?<!^)(?=[A-Z])', ' ', state)

    # import data 
    data = pd.read_csv("./data/all-states-2010-2024.csv", skiprows=1) 

    ## Preprocessing

    # create Week_start column
    data["Week_start"] = pd.to_datetime(
        data["YEAR"].astype(str) + data["WEEK"].astype(str) + "-1", format="%G%V-%u"
    ).dt.normalize()

    # get rows only for state of interest
    filtered_data = data[data["REGION"] == state]

    # filter %UNWEIGHTED ILI column and make numeric
    filtered_data = filtered_data[["Week_start", "%UNWEIGHTED ILI"]]
    filtered_data["%UNWEIGHTED ILI"] = pd.to_numeric(filtered_data["%UNWEIGHTED ILI"])

    # divide %UNWEIGHTED ILI column by 100 (normalise between 0 and 1)
    filtered_data["%UNWEIGHTED ILI"] = filtered_data["%UNWEIGHTED ILI"] / 100

    # drop duplicates and reset index
    filtered_data = filtered_data.drop_duplicates(subset=["Week_start"], keep="first")
    filtered_data = filtered_data.reset_index(drop=True)

    # use split_week to create training and test sets
    split_week_dt = pd.to_datetime(split_week)
    train_data = filtered_data[filtered_data["Week_start"] < split_week_dt]
    test_data = filtered_data[filtered_data["Week_start"] >= split_week_dt]

    ## Forecasting

    quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    timegpt_fcst_df = nixtla_client.forecast(
        df=train_data, 
        quantiles=quantiles,
        h=52, # forecast for 52 weeks ahead
        freq='W-MON', 
        time_col='Week_start', 
        target_col='%UNWEIGHTED ILI'
    )

    # formatting forecast outputs
    timegpt_fcst_df['State'] = state
    timegpt_fcst_df['Split Week'] = split_week
    timegpt_fcst_df['Model'] = 'TimeGPT'
    timegpt_fcst_df['Prediction Horizon'] = timegpt_fcst_df.index + 1

    timegpt_fcst_df['Split Week'] = pd.to_datetime(timegpt_fcst_df['Split Week'])
    timegpt_fcst_df['Year'] = timegpt_fcst_df['Split Week'].dt.year

    timegpt_fcst_df.rename(columns={'Week_start': 'Prediction Week'}, inplace=True)

    # add real data to forecasts df 
    real_data = test_data.head(52).copy()
    timegpt_fcst_df['Real'] = real_data['%UNWEIGHTED ILI'].values

    # reorder columns 
    timegpt_fcst_df = timegpt_fcst_df[['State', 'Year', 'Model', 'Split Week', 'Prediction Week', 'Prediction Horizon', 'Real', 'TimeGPT',  
                            'TimeGPT-q-5', 'TimeGPT-q-10', 'TimeGPT-q-15', 'TimeGPT-q-20', 'TimeGPT-q-25', 'TimeGPT-q-30', 'TimeGPT-q-35', 'TimeGPT-q-40',
                            'TimeGPT-q-45', 'TimeGPT-q-50', 'TimeGPT-q-55', 'TimeGPT-q-60', 'TimeGPT-q-65', 'TimeGPT-q-70', 'TimeGPT-q-75', 'TimeGPT-q-80',
                            'TimeGPT-q-85', 'TimeGPT-q-90', 'TimeGPT-q-95']]

    ## Store forecasts and evaluation metrics

    horizons = [1,4,13,26,52]

    for horizon in horizons:
        # location to store forecasts
        output_dir = f"out_ili_%/point_prediction/{state}/{horizon}week/forecasts" #
        output_file = f"{output_dir}/{split_week}.csv"
        os.makedirs(output_dir, exist_ok=True)

        # extract first n weeks of forecasts
        forecasts = timegpt_fcst_df.head(horizon).copy()

        # save forecasts as csv 
        forecasts.to_csv(output_file, index=False)

        # calculate evaluation metrics - MAE, MAPE, sMAPE, RMSE, MASE
        real_last = forecasts['Real'].iloc[-1] # point prediction (only compare last row)
        timegpt_last = forecasts['TimeGPT'].iloc[-1]

        mae = abs(real_last - timegpt_last)
        mape = abs((real_last - timegpt_last) / real_last) * 100
        smape = sMAPE(timegpt_last, real_last)
        rmse = np.sqrt((real_last - timegpt_last)**2)

        mase = mean_absolute_scaled_error(
            y_true=forecasts['Real'],
            y_pred=forecasts['TimeGPT'],
            y_train=train_data['%UNWEIGHTED ILI']
        )

        evaluation_metrics = {
            'Split Week': split_week,
            'MAE': mae,
            'MAPE': mape,
            'sMAPE': smape,
            'RMSE': rmse,
            'MASE': mase
        }

        # location to store evaluation metrics
        eval_dir = f"out_ili_%/point_prediction/{state}/{horizon}week/evaluation"
        eval_file = f"{eval_dir}/{split_week}.csv"
        os.makedirs(eval_dir, exist_ok=True)

        # save evaluations as csv
        eval_df = pd.DataFrame(evaluation_metrics, index=[0])
        eval_df.to_csv(eval_file, index=False)

    ## Winter analysis 

    # define winter months (November, December, January, February, March, April)
    winter_months = [11, 12, 1, 2, 3, 4]

    for horizon in [1,4,13]:
        # location to store forecasts
        output_dir = f"out_ili_%/point_prediction/winter/{state}/{horizon}week/forecasts"
        output_file = f"{output_dir}/{split_week}.csv"
        os.makedirs(output_dir, exist_ok=True)

        # extract first n weeks of forecasts
        forecasts = timegpt_fcst_df.head(horizon).copy()

        # extract forecasts for winter months 
        forecasts['Split Week'] = pd.to_datetime(forecasts['Split Week'])
        forecasts['Month'] = forecasts['Split Week'].dt.month
        filtered_data = forecasts[forecasts['Month'].isin(winter_months)]
        filtered_data = filtered_data.drop(columns=['Month'])

        # save forecasts and do evaluation if filtered df is not empty
        if not filtered_data.empty:
            # save forecasts as csv 
            filtered_data.to_csv(output_file, index=False)

            # calculate evaluation metrics - MAE, MAPE, sMAPE, RMSE, MASE
            real_last = forecasts['Real'].iloc[-1] # point prediction (only compare last row)
            timegpt_last = forecasts['TimeGPT'].iloc[-1]

            mae = abs(real_last - timegpt_last)
            mape = abs((real_last - timegpt_last) / real_last) * 100
            smape = sMAPE(timegpt_last, real_last)
            rmse = np.sqrt((real_last - timegpt_last)**2)

            mase = mean_absolute_scaled_error(
                y_true=forecasts['Real'],
                y_pred=forecasts['TimeGPT'],
                y_train=train_data['%UNWEIGHTED ILI']
            )

            evaluation_metrics = {
                'Split Week': split_week,
                'MAE': mae,
                'MAPE': mape,
                'sMAPE': smape,
                'RMSE': rmse,
                'MASE': mase
            }

            # location to store evaluation metrics
            eval_dir = f"out_ili_%/point_prediction/winter/{state}/{horizon}week/evaluation"
            eval_file = f"{eval_dir}/{split_week}.csv"
            os.makedirs(eval_dir, exist_ok=True)

            # save evaluations as csv
            eval_df = pd.DataFrame(evaluation_metrics, index=[0])
            eval_df.to_csv(eval_file, index=False)
        
if __name__ == "__main__":
    main()