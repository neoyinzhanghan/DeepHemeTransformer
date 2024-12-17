import os
import pandas as pd
from LLRunner.read.SST import sst
from tqdm import tqdm

diff_data_csv_path = "/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv"

# open the csv file
df = pd.read_csv(diff_data_csv_path)

# get the accession_number column as a list of strings
accession_numbers = df["accession_number"].tolist()

dx_df_dict = {
    "accession_number": [],
    "dx": [],
    "subdx": [],
}

for accession_number in tqdm(accession_numbers, desc="Looking for dxes"):

    try:
        dx, subdx = sst.get_dx(accession_number)
        dx_df_dict["accession_number"].append(accession_number)
        dx_df_dict["dx"].append(dx)
        dx_df_dict["subdx"].append(subdx)
    except Exception as e:
        pass

dx_df = pd.DataFrame(dx_df_dict)

# only keeps the rows where the dx column is "Normal BMA" or "Plasma cell myeloma"
dx_df = dx_df[(dx_df["dx"] == "Normal BMA") | (dx_df["dx"] == "Plasma cell myeloma")]

dx_df.to_csv("/media/hdd3/neo/dx_data_test.csv", index=False)
