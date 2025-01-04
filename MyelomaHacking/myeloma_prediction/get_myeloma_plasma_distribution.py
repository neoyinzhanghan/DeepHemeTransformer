import pandas as pd
from LLRunner.read.SST import sst
from get_copath_data import get_path_data, get_diff

df = sst.df

# get the accession_number column of df as a list of strings
accession_numbers = df["accession_number"].tolist()

print(f"Founds {len(accession_numbers)} accession numbers")

print("Extracting Path df")
path_df = get_path_data(accession_numbers)
print("Extracting Diff df")
diff_df = get_diff(path_df)

path_df.to_csv("/media/hdd3/neo/myeloma_path_data_processed.csv", index=False)
diff_df.to_csv("/media/hdd3/neo/myeloma_diff_data_processed.csv", index=False)