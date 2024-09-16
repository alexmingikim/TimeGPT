import os
import pandas as pd

# directory containing the CSV files
input_directory = './out_ili_%/California/13week/forecasts' #
output_file = './out_ili_%/California/13week/joined_forecasts.csv' #

# list to store data from each CSV file
all_data = []

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_directory, filename)
        
        df = pd.read_csv(filepath)
        
        # extract last row from relevant columns
        last_row = df[['Split Week', 'Prediction Week', 'Prediction Horizon', 'Real', 'TimeGPT']].iloc[-1]
        
        all_data.append(last_row)

joined_df = pd.DataFrame(all_data)

# sort df by 'Split Week'
joined_df = joined_df.sort_values(by='Split Week')

joined_df.to_csv(output_file, index=False)