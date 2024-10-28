import os
import pandas as pd
from tqdm import tqdm
from BMAassumptions import BMA_final_classes

metadata_path = "/media/hdd3/neo/features_metadata.csv"
new_metadata_path = "/media/hdd3/neo/labelled_features_metadata.csv"
differential_data_path = "/media/hdd3/neo/test_diff_data.csv"


new_metadata_dict = {
    "wsi_name": [],
    "accession_number": [],
    "result_dir_name": [],
    "features_path": [],
}

for diff_class in BMA_final_classes:
    new_metadata_dict[diff_class] = []

# traverse through the rows of metadata_path
metadata = pd.read_csv(metadata_path)
differential_df = pd.read_csv(differential_data_path)

for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    wsi_name = row["wsi_name"]
    result_dir_name = row["result_dir_name"]
    features_path = row["features_path"]

    accession_number = wsi_name.split(";")[0]

    # look for the rows in the differential data file where the specnum_formatted column matches the accession number
    diff_data_row = differential_df[
        differential_df["specnum_formatted"] == accession_number
    ]

    if len(diff_data_row) != 1:
        print(
            f"UserWarning: Exactly only one row should match the accession number {accession_number}. Instead, {len(diff_data_row)} rows matched."
        )

    for diff_class in BMA_final_classes:
        new_metadata_dict[diff_class].append(diff_data_row[diff_class])

    new_metadata_dict["wsi_name"].append(wsi_name)
    new_metadata_dict["accession_number"].append(accession_number)
    new_metadata_dict["result_dir_name"].append(result_dir_name)
    new_metadata_dict["features_path"].append(features_path)

new_metadata_df = pd.DataFrame(new_metadata_dict)

new_metadata_df.to_csv(new_metadata_path, index=False)
