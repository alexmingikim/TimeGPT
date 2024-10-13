import os
from dotenv import load_dotenv
import pandas as pd

# load .env file
load_dotenv()

# read the start and end dates and states of interest from the environment variables
start_date = os.getenv("START_DATE")
end_date = os.getenv("END_DATE")
states = os.getenv("STATES").split(",")

# generate date range for each week between start_date and end_date
weeks = pd.date_range(start=start_date, end=end_date, freq="W-MON")

# generate command strings for each week
commands = []
for state in states:
    for date in weeks:
        commands.append(
            f"srun --unbuffered python timegpt_script_%unweightedILI_zeroshot.py {state} {date.strftime('%Y-%m-%d')}" #
        )

# save command strings to a text file with an empty line at the end
with open("execs/train_models", "w") as file:
    file.write("\n".join(commands) + "\n")
