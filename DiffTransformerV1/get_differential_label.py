import pandas as pd
from get_copath_data import get_diff, get_path_data

metadata_path = "/media/hdd3/neo/DiffTransformerV1DataMini/wsi_metadata.csv"
metadata = pd.read_csv(metadata_path)

# get the accession_numbers as a list of strings
accession_numbers = metadata["accession_number"].tolist()

# print the first 5 accession numbers
print(accession_numbers[:5])

# print the length of the accession_numbers list
print(f"Number of accession numbers: {len(accession_numbers)}")

path_df = get_path_data(accession_numbers)

# print the number of rows in the path data
print(f"Number of rows in path data: {len(path_df)}")

diff = get_diff(path_df)

# print the number of rows in the differential data
print(f"Number of rows in differential data: {len(diff)}")

# save the differential data to a csv file
diff.to_csv("/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv", index=False)
