import os
import pandas as pd
from tqdm import tqdm
from get_copath_data import get_path_data, get_diff
from LLRunner.read.SST import sst

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


# get the wsi_name column as a list of strings
wsi_names = pipeline_run_history["wsi_name"].tolist()
accession_numbers = []

for wsi_name in tqdm(wsi_names, desc="Extracting Accession Numbers"):
    accession_numbers.append(wsi_name.split(";")[0])

print("Extracting Path df")
df = get_path_data(accession_numbers)

print("Extracting Diff df")
diff_df = get_diff(df)

# save the df and diff_df to csv files at /media/hdd3/neo/path_data_processed.csv and /media/hdd3/neo/diff_data_processed.csv
df.to_csv("/media/hdd3/neo/path_data_processed.csv", index=False)
diff_df.to_csv("/media/hdd3/neo/diff_data_processed.csv", index=False)