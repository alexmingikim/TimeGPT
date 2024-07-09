import os
import pandas as pd

# Define states and forecast horizons
# TODO: specify states and horizons as environmental variables
states = ["Texas"] #
horizons = ['1week', '4week', '13week', '26week', '52week'] #

# Loop over each forecast horizon
for state in states: 
    for horizon in horizons:
        root_dir = f'output/%UNWEIGHTED ILI/hyperparameter_tuning/100/{state}' #
        eval_dir = os.path.join(root_dir, horizon, 'evaluation')
        
        # Read all csv files in evaluation directory
        metrics = []
        for file in os.listdir(eval_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(eval_dir, file)
                df = pd.read_csv(file_path)
                metrics.append(df[['MAE', 'MAPE', 'sMAPE', 'RMSE', 'MASE']]) #
        
        # Concatenate all dataframes
        if metrics:
            all_metrics = pd.concat(metrics)
            
            # Compute average of the metrics
            avg_metrics = all_metrics.mean().to_frame().T
            
            # Write summary to csv file
            summary_file_path = os.path.join(root_dir, horizon, 'summary_evaluation.csv')
            avg_metrics.to_csv(summary_file_path, index=False)