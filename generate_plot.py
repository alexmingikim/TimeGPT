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

                # Extract MAPE value (assuming it's in the first row and MAPE column)
                mape_value = df.loc[0, 'MAPE'] #
                
                # Extract RMSE value (assuming it's in the first row and RMSE column)
                rmse_value = df.loc[0, 'RMSE'] #
                
                # Append the data to the list
                data.append({'WEEK_START': week_start, 'RMSE': rmse_value, 'MAPE': mape_value})

        # Create DataFrame
        result_df = pd.DataFrame(data)

        # Convert WEEK_START to datetime for plotting
        result_df['WEEK_START'] = pd.to_datetime(result_df['WEEK_START'])

        # Sort DataFrame by WEEK_START
        result_df = result_df.sort_values('WEEK_START')

        # Save DataFrame to a CSV file
        output_path = os.path.join(directory, f'summmary_evaluation_MAPE+RMSE.csv')
        result_df.to_csv(output_path, index=False)

        # Plot DataFrame
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot RMSE
        ax1.set_xlabel('Week Start')
        ax1.set_ylabel('RMSE', color='tab:blue')
        ax1.plot(result_df['WEEK_START'], result_df['RMSE'], color='tab:blue', marker='o', label='RMSE')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create second y-axis to plot MAPE
        ax2 = ax1.twinx()
        ax2.set_ylabel('MAPE', color='tab:red')
        ax2.plot(result_df['WEEK_START'], result_df['MAPE'], color='tab:red', marker='x', label='MAPE')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Add title and grid
        plt.title(f'RMSE and MAPE over Time for {state} - {horizon}')
        fig.tight_layout()
        plt.grid(True)

        # Show plot
        plt.show()
