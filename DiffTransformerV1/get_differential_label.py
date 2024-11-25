import pandas as pd
from get_copath_data import get_diff, get_path_data

metadata_path = "/media/hdd3/neo/DiffTransformerV1DataMini/wsi_metadata.csv"
metadata = pd.read_csv(metadata_path)

# get the accession_numbers as a list of strings
accession_numbers = metadata["accession_number"].tolist()

path_df = get_path_data(accession_numbers)
diff = get_diff(path_df)

# save the differential data to a csv file
diff.to_csv("/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv", index=False)
