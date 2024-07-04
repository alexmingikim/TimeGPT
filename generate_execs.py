import pandas as pd

# Variable definition
start_date = "2017-06-01"
end_date = "2019-06-01"
states = ["California", "Minnesota", "Nevada", "Utah", "Virginia", "Texas", "Wyoming"] #

# Generate date range for each week in between start and end dates
weeks = pd.date_range(start=start_date, end=end_date, freq="W-MON")

# Generate command strings for each date
commands = []
for state in states:
    for date in weeks:
        commands.append(
            f"srun --unbuffered python timegpt_script.py {state} {date.strftime('%Y-%m-%d')}" #
        )

# Save command strings to a text file with an empty line at the end
with open("execs/train_models", "w") as file:
    file.write("\n".join(commands) + "\n")
