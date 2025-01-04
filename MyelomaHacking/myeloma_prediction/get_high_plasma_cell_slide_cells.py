import os
import pandas as pd
from tqdm import tqdm

result_dir_path = "/media/hdd4/neo/results_dir"
pipeline_run_history_path = "/media/hdd4/neo/results_dir/pipeline_run_history.csv"


# find the subdirs in the result_dir_path that does not start with ERROR_

subdirs = os.listdir(result_dir_path)

# print the number of subdirs found
print(f"Number of subdirs found: {len(subdirs)}")

# open the pipeline_run_history.csv file
pipeline_run_history = pd.read_csv(pipeline_run_history_path)

# print how many rows in total and how many rows have error equal to False
print(f"Total rows: {len(pipeline_run_history)}")
print(f"Rows with error=False: {len(pipeline_run_history[pipeline_run_history['error'] == False])}")