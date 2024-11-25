import pandas as pd
from get_copath_data import get_diff

metadata_path = "/media/hdd3/neo/DiffTransformerV1DataMini/wsi_metadata.csv"
metadata = pd.read_csv(metadata_path)

# get the accession_numbers as a list of strings
accession_numbers = metadata["accession_number"].tolist()

diff = get_diff(accession_numbers)

print(type(diff))
