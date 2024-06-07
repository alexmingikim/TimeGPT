import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from nixtla import NixtlaClient
nixtla_client = NixtlaClient()
from sys import *
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Get arguments
state = argv[1]
split_week = datetime.strptime(argv[2], "%Y-%m-%d").date()

# Import data 
df = pd.read_csv("../data/all-states-2010-2024.csv", skiprows=1)

### Preprocessing

# Create WEEK_START column
df["WEEK_START"] = pd.to_datetime(
    df["YEAR"].astype(str) + df["WEEK"].astype(str) + "-1", format="%G%V-%u"
).dt.normalize()

# Get only rows for state of interest
filtered_df = df[df["REGION"] == state]

# Drop irrelevant columns
filtered_df = filtered_df[["WEEK_START", "ILITOTAL"]]

# Drop duplicates
filtered_df = filtered_df.drop_duplicates(subset=["WEEK_START"], keep="first")

# Reset index
filtered_df = filtered_df.reset_index(drop=True)

# Make 'ILITOTAL' column numeric
filtered_df["ILITOTAL"] = pd.to_numeric(filtered_df["ILITOTAL"])

# Convert split_week to datetime
split_week_dt = pd.to_datetime(split_week)

# Use split week to create training set
train_data = filtered_df[filtered_df["WEEK_START"] < split_week_dt]

# Create test set
test_data = filtered_df[filtered_df["WEEK_START"] >= split_week_dt]

### Forecasting
quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

timegpt_fcst_df = nixtla_client.forecast(
    df=train_data, 
    quantiles=quantiles,
    h=52, 
    freq='W-MON', 
    time_col='WEEK_START', 
    target_col='ILITOTAL'
)

### Evaluation

# Merge forecasted values with test set for evaluation
test_data = test_data.head(52)
test_data['TimeGPT'] = timegpt_fcst_df['TimeGPT'].values

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(test_data['ILITOTAL'], test_data['TimeGPT']))
mae = mean_absolute_error(test_data['ILITOTAL'], test_data['TimeGPT'])
mape = mean_absolute_percentage_error(test_data['ILITOTAL'], test_data['TimeGPT'])

# Define evaluation metrics
evaluation_metrics = {
    'Split_week': split_week,
    'RMSE': rmse,
    'MAE': mae,
    'MAPE': mape
}

### Save predictions and evaluations

# Define path
output_dir = f"output/{state}/predictions"
output_file = f"{output_dir}/{split_week}_predictions.csv"
eval_dir = f"output/{state}/evaluation"
eval_file = f"{eval_dir}/evaluation.csv"

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Save prediction DataFrame to CSV
timegpt_fcst_df.to_csv(output_file, index=False)

# Check if evaluation file exists
if os.path.exists(eval_file):
    # Read existing evaluation file
    eval_df = pd.read_csv(eval_file)
    # Append new evaluation metrics as new row
    eval_df = eval_df.append(evaluation_metrics, ignore_index=True)
else:
    # Create new DataFrame with evaluation metrics
    eval_df = pd.DataFrame(evaluation_metrics, index=[0])

# Save evaluation DataFrame to CSV
eval_df.to_csv(eval_file, index=False)
