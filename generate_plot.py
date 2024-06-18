import os
import pandas as pd
import matplotlib.pyplot as plt

states = ['Texas'] #
horizons = ['4week'] #

for state in states: 
    for horizon in horizons:
        # Directory containing the CSV files
        directory = f'output/{state}/{horizon}/evaluation'

        # Initialise empty list to store data
        data = []

        # Iterate over each CSV file in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                # Extract week_start from filename (i.e. remove the '.csv' part)
                week_start = filename[:-4]
                
                # Construct full path to file
                filepath = os.path.join(directory, filename)
                
                # Read CSV file into a DataFrame
                df = pd.read_csv(filepath)
                
                # Extract RMSE value (assuming it's in the first row and RMSE column)
                rmse_value = df.loc[0, 'RMSE'] #
                
                # Append the data to the list
                data.append({'WEEK_START': week_start, 'TimeGPT': rmse_value})

        # Create DataFrame
        result_df = pd.DataFrame(data)

        # Convert WEEK_START to datetime for plotting
        result_df['WEEK_START'] = pd.to_datetime(result_df['WEEK_START'])

        # Sort DataFrame by WEEK_START
        result_df = result_df.sort_values('WEEK_START')

        # Save the DataFrame to a CSV file
        output_path = os.path.join(directory, f'summmary_evaluation_{horizon}.csv')
        result_df.to_csv(output_path, index=False)

        # Plot the DataFrame
        plt.figure(figsize=(10, 5))
        plt.plot(result_df['WEEK_START'], result_df['TimeGPT'], marker='o')
        plt.xlabel('Week Start')
        plt.ylabel('RMSE (TimeGPT)')
        plt.title('RMSE over Time')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
