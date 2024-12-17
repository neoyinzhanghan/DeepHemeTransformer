import os
import pandas as pd
from LLRunner.read.SST import sst
from tqdm import tqdm

diff_data_csv_path = "/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv"

# open the csv file
df = pd.read_csv(diff_data_csv_path)

# get the accession_number column as a list of strings
accession_numbers = df["accession_number"].tolist()

have_dx = 0
no_dx = 0

for accession_number in tqdm(accession_numbers, desc="Looking for dxes"):

    try:
        dx, subdx = sst.get_dx(accession_number)
        have_dx += 1
    except Exception as e:
        dx = "None"
        subdx = "None"
        no_dx += 1

print(f"Have dx: {have_dx}")
print(f"No dx: {no_dx}")
