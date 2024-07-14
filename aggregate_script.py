import os
import pandas as pd

# TODO: include real data

states = ["California", "Texas"]
horizons = ["1week", "4week"]
winter_months = [11, 12, 1, 2, 3, 4]

for state in states:
    for horizon in horizons:
        base_dir = f'output/%UNWEIGHTED ILI/zeroshot/{state}/{horizon}'
        root_dir = f'output/%UNWEIGHTED ILI/zeroshot/{state}/{horizon}/forecasts'
        aggregated_df = pd.DataFrame()

        for filename in os.listdir(root_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(root_dir, filename)
                df = pd.read_csv(file_path)
                aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)

        # Convert 'Split_week' to datetime and sort by date
        aggregated_df['Split_week'] = pd.to_datetime(aggregated_df['Split_week'])
        aggregated_df = aggregated_df.sort_values(by='Split_week')

        # Extract forecasts for winter months
        aggregated_df['Month'] = aggregated_df['Split_week'].dt.month
        filtered_forecasts = aggregated_df[aggregated_df['Month'].isin(winter_months)]
        filtered_forecasts = filtered_forecasts.drop(columns=['Month'])

        # Reset index after filtering
        filtered_forecasts.reset_index(drop=True, inplace=True)

        # Save filtered forecasts as csv
        output_file_path = os.path.join(base_dir, 'aggregated_forecasts_winter.csv')
        filtered_forecasts.to_csv(output_file_path, index=False)
