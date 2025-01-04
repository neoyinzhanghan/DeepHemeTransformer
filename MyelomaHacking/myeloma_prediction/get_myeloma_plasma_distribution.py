import pandas as pd
from LLRunner.read.SST import sst
from get_copath_data import get_path_data, get_diff

sst.load_dataframe()
df = sst.df

# get the accession_number column of df as a list of strings
accession_numbers = df["Accession Number"].tolist()

print(f"Founds {len(accession_numbers)} accession numbers")

myeloma_accession_numbers = []
for accession_number in accession_numbers:
    dx, subdx = sst.get_dx(accession_number)

    if dx == "Plasma cell myeloma":
        myeloma_accession_numbers.append(accession_number)

print(f"Found {len(myeloma_accession_numbers)} myeloma accession numbers")

print("Extracting Path df")
path_df = get_path_data(myeloma_accession_numbers)
print("Extracting Diff df")
diff_df = get_diff(path_df)

path_df.to_csv("/media/hdd3/neo/myeloma_path_data_processed.csv", index=False)
diff_df.to_csv("/media/hdd3/neo/myeloma_diff_data_processed.csv", index=False)