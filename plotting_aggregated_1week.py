import os
import pandas as pd

# TODO: include real data

states = ["California", "Texas"]

for state in states:
    root_dir = f'output/%UNWEIGHTED ILI/zeroshot/{state}/1week/forecasts'
    aggregated_df = pd.DataFrame()

    for filename in os.listdir(root_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(root_dir, filename)
            df = pd.read_csv(file_path)
            aggregated_df = pd.concat([aggregated_df, df])

    # Convert 'Split_week' to datetime and sort by date
    aggregated_df['Split_week'] = pd.to_datetime(aggregated_df['Split_week'])
    aggregated_df = aggregated_df.sort_values(by='Split_week')

    # Reset index after sorting
    aggregated_df.reset_index(drop=True, inplace=True)

    # Save aggregated forecasts as csv
    output_file_path = os.path.join(root_dir, 'aggregated_forecasts.csv')
    aggregated_df.to_csv(output_file_path, index=False)
