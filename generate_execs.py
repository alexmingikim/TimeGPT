import pandas as pd

# Generate date range for each week of 2017 and 2019
weeks = pd.date_range(start="2017-06-01", end="2019-06-01", freq="W")
states = ["Georgia"]

# Generate command strings for each date
commands = []
for state in states:
    for date in weeks:
        commands.append(
            f"srun --unbuffered python timegpt_script.py {state} {date.strftime('%Y-%m-%d')}"
        )

# Save command strings to a text file
with open("execs/train_models", "w") as file:
    file.write("\n".join(commands))
