import os
import pandas as pd
import numpy as np

# Function to compute IQR
def compute_iqr(column):
    q3, q1 = np.percentile(column, [75 ,25])
    return q3 - q1

# Alternative function to compute IQR
# source: https://stackoverflow.com/questions/51943661/is-scipy-stats-doing-wrong-calculation-for-iqr

# def compute_iqr(m):
#     m = np.array(m)
#     n = m.size//2
#     m_ = np.partition(m.ravel(), n + 1)
#     return np.median(m_[n + m.size%2:]) - np.median(m_[:n])

# Define states and forecast horizons
# TODO: specify states and horizons as environmental variables
states = ["California", "Texas"] #
horizons = ['13week'] #

# Loop over each forecast horizon
for state in states: 
    for horizon in horizons:
        root_dir = f'output/%UNWEIGHTED ILI/hyperparameter_tuning/100/{state}' #
        eval_dir = os.path.join(root_dir, horizon, 'evaluation')
        
        # Read all csv files in evaluation directory
        metrics = []
        for file in os.listdir(eval_dir):
            if file.endswith('.csv') and (file.startswith('2017-11') or file.startswith('2017-12') or file.startswith('2018-01')):
                file_path = os.path.join(eval_dir, file)
                df = pd.read_csv(file_path)
                metrics.append(df[['MAE', 'MAPE', 'sMAPE', 'RMSE', 'MASE']]) #

        # Concatenate all dataframes
        if metrics:
            all_metrics = pd.concat(metrics)
            
            # Compute average of the metrics
            avg_metrics = all_metrics.mean().to_frame().T
            
            # Compute IQR of the metrics
            iqr_metrics = all_metrics.apply(compute_iqr).to_frame().T
            
            # Combine average and IQR metrics
            summary_metrics = pd.concat([avg_metrics, iqr_metrics])
            summary_metrics.index = ['Average', 'IQR']
            
            # Write summary to csv file
            summary_file_path = os.path.join(root_dir, horizon, 'summary_evaluation_winter.csv')
            summary_metrics.to_csv(summary_file_path)