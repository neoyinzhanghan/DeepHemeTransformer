import os
import pandas as pd
from tqdm import tqdm
from BMAassumptions import BMA_reported_classes, BMA_final_classes

metadata_path = "/media/hdd3/neo/DeepHemeTransformerData/features_metadata.csv"
new_metadata_path = (
    "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
)
differential_data_path = "/media/hdd3/neo/differential_data_2024-10-10.csv"


new_metadata_dict = {
    "wsi_name": [],
    "accession_number": [],
    "result_dir_name": [],
    "features_path": [],
}

for diff_class in BMA_reported_classes:
    new_metadata_dict[diff_class] = []

# traverse through the rows of metadata_path
metadata = pd.read_csv(metadata_path)
differential_df = pd.read_csv(differential_data_path)

# make sure the specnum_formatted column is a string

num_problematic = 0

for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    wsi_name = row["wsi_name"]
    result_dir_name = row["result_dir_name"]
    features_path = row["features_path"]

    accession_number = wsi_name.split(";")[0]

    # look for the rows in the differential data file where the specnum_formatted column matches the accession number
    diff_data_row = differential_df[
        differential_df["specnum_formatted"] == accession_number
    ]

    if len(diff_data_row) == 0:
        print(
            f"UserWarning: No differential data found for {accession_number}. Skipping."
        )
        num_problematic += 1
        continue

    if len(diff_data_row) > 1:
        diff_data_row = diff_data_row.iloc[0]

    for diff_class in BMA_reported_classes:

        val = diff_data_row[diff_class]

        # get the value, right now it is a pandas.core.series.Series
        # we need to get the actual value
        val = val.values[0]

        # now it is numpy.float64, we want to convert it to a float, and if it is nan, convert it to 0
        if pd.isna(val):
            val = 0.0
        else:
            val = float(val)

        new_metadata_dict[diff_class].append(val)

    new_metadata_dict["wsi_name"].append(wsi_name)
    new_metadata_dict["accession_number"].append(accession_number)
    new_metadata_dict["result_dir_name"].append(result_dir_name)
    new_metadata_dict["features_path"].append(features_path)

new_metadata_df = pd.DataFrame(new_metadata_dict)

# sum the blasts and blast_equivalents column together into a new column named "blasts and blast equivalents"
new_metadata_df["blasts and blast-equivalents"] = (
    new_metadata_df["blasts"] + new_metadata_df["blast-equivalents"]
)

# delete the blasts and blast_equivalents columns
new_metadata_df.drop(columns=["blasts", "blast-equivalents"], inplace=True)

# sum the colums in BMA_final_classes which is a list of strings and assign to a new column named "total"
new_metadata_df["total"] = new_metadata_df[BMA_final_classes].sum(axis=1)

# find how many rows have a total value < 90
num_total_less_than_90 = len(new_metadata_df[new_metadata_df["total"] < 90])

# remove all rows where the total value is < 90
new_metadata_df = new_metadata_df[new_metadata_df["total"] >= 90]

print(f"Number of rows with total < 90: {num_total_less_than_90}")
print(f"Number of problematic rows: {num_problematic}")
print(
    f"Number of non-problematic rows: {len(metadata) - num_problematic - num_total_less_than_90}"
)
new_metadata_df.to_csv(new_metadata_path, index=False)
