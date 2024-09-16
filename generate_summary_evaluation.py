import os
import pandas as pd
import numpy as np

def compute_iqr(column):
    q3, q1 = np.percentile(column, [75 ,25])
    return q3 - q1

# define states and forecast horizons
# TODO: specify states and horizons as environmental variables
states = ["California"] #
horizons = ['1', '4', '13', '26', '52'] #

for state in states: 
    for horizon in horizons:
        root_dir = f'out_ili_%/{state}' 
        eval_dir = os.path.join(root_dir, str(horizon)+'week', 'evaluation', 'averaged') # 
        # eval_dir = os.path.join(root_dir, str(horizon)+'week', 'evaluation', 'point_prediction') #
        
        # read all csv files in evaluation directory
        metrics = []
        for file in os.listdir(eval_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(eval_dir, file)
                df = pd.read_csv(file_path)
                metrics.append(df[['MAE', 'MAPE', 'sMAPE', 'RMSE', 'MASE']]) 

        # concatenate all dataframes
        if metrics:
            all_metrics = pd.concat(metrics)
            # compute average
            avg_metrics = all_metrics.mean().to_frame().T
            # compute IQR
            iqr_metrics = all_metrics.apply(compute_iqr).to_frame().T
            # combine average and IQR
            summary_metrics = pd.concat([avg_metrics, iqr_metrics])
            summary_metrics.index = ['Average', 'IQR']
            # write to csv
            summary_file_path = os.path.join(root_dir, str(horizon)+'week', 'evaluation', 'summary_averaged.csv') #
            # summary_file_path = os.path.join(root_dir, str(horizon)+'week', 'evaluation', 'summary_point_prediction.csv') #
            summary_metrics.to_csv(summary_file_path)
    
    ## Winter analysis
    for horizon in [1,4,13]:
        root_dir = f'out_ili_%/winter_analysis/{state}'
        eval_dir = os.path.join(root_dir, str(horizon)+'week', 'evaluation', 'averaged') #
        # eval_dir = os.path.join(root_dir, str(horizon)+'week', 'evaluation', 'point_prediction') # 
        
        # read all csv files in evaluation directory
        metrics = []
        for file in os.listdir(eval_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(eval_dir, file)
                df = pd.read_csv(file_path)
                metrics.append(df[['MAE', 'MAPE', 'sMAPE', 'RMSE', 'MASE']])

        # concatenate all dataframes
        if metrics:
            all_metrics = pd.concat(metrics)
            # compute average
            avg_metrics = all_metrics.mean().to_frame().T
            # compute IQR
            iqr_metrics = all_metrics.apply(compute_iqr).to_frame().T
            # combine average and IQR
            summary_metrics = pd.concat([avg_metrics, iqr_metrics])
            summary_metrics.index = ['Average', 'IQR']
            # write to csv
            summary_file_path = os.path.join(root_dir, str(horizon)+'week', 'evaluation', 'summary_averaged.csv') #
            # summary_file_path = os.path.join(root_dir, str(horizon)+'week', 'evaluation', 'summary_point_prediction.csv') #
            summary_metrics.to_csv(summary_file_path)