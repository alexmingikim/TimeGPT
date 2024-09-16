import pandas as pd

# Define start and end dates
start_date = "2017-06-01"
end_date = "2019-06-01"

# Generate date range for each week of 2017 and 2019
weeks = pd.date_range(start=start_date, end=end_date, freq="W-MON")
states = ["California"] #

# Generate command strings for each date
commands = []
for state in states:
    for date in weeks:
        commands.append(
            f"srun --unbuffered python timegpt_script_%unweightedILI_zeroshot.py {state} {date.strftime('%Y-%m-%d')}" #
        )

# Save command strings to a text file with an empty line at the end
with open("execs/train_models", "w") as file:
    file.write("\n".join(commands) + "\n")
