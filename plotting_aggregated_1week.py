import os
import pandas as pd

states = ["California", "Texas"]

for state in states:
    root_dir = f'output/%UNWEIGHTED ILI/zeroshot/{state}/1week/forecasts' #

    for filename in os.listdir(root_dir):
        if filename.endswith(".csv"):
            aggregated_df = pd.DataFrame()
            file_path = os.path.join(root_dir, filename)
            df = pd.read_csv(file_path)
            aggregated_df = pd.concat([aggregated_df, df])

    # Reset index after concatenation
    aggregated_df.reset_index(drop=True, inplace=True)

    # Save aggregated forecasts as csv
    output_file_path = os.path.join(root_dir, 'aggregated_forecasts.csv')
    aggregated_df.to_csv(output_file_path, index=False)
