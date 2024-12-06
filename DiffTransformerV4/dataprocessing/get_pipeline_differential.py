import os
import time
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from BMAassumptions import (
    removed_classes,
    omitted_classes,
    non_removed_classes,
    BMA_final_classes,
    differential_group_dict,
    cellnames,
)

data_dir = "/media/hdd3/neo/DiffTransformerV1Data3000"
diff_data_path = "/media/hdd3/neo/DiffTransformerV1Data3000/diff_data.csv"

cellnames = [
    "B1",
    "B2",
    "E1",
    "E4",
    "ER1",
    "ER2",
    "ER3",
    "ER4",
    "ER5",
    "ER6",
    "L2",
    "L4",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "MO2",
    "PL2",
    "PL3",
    "U1",
    "U4",
]

# get the list of all subdirectories in the data directory
all_subdirs = [
    d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
]

# onlu keep the one that starts with BMA and PBS
result_dirs = [d for d in all_subdirs if "BMA-diff" in d or "PBS-diff" in d]

all_result_dir_paths = [os.path.join(data_dir, d) for d in result_dirs]

num_errors = 0
num_dirs = len(all_result_dir_paths)
non_error_dirs = []
error_dirs = []

all_cell_paths = []
for result_dir_path in tqdm(all_result_dir_paths, desc="Filtering out error dirs:"):
    # check if the result_dir_path contains a file called "error.txt"
    if not os.path.exists(os.path.join(result_dir_path, "error.txt")):
        non_error_dirs.append(result_dir_path)

    else:
        error_dirs.append(result_dir_path)
        num_errors += 1

print(f"Number of error dirs: {num_errors}")
print(f"Number of non-error dirs: {len(non_error_dirs)}")

nonerror_df_dict = {
    "result_dir_name": [],
    "num_cells": [],
    "num_regions": [],
    "specimen_type": [],
}

for cell_name in cellnames:
    nonerror_df_dict[str("num" + cell_name)] = []

for non_error_dir in tqdm(
    non_error_dirs,
    desc="Gathering aggregate pipeline results for non-error directories",
):
    specimen_type = "BMA" if "BMA-diff" in non_error_dir else "PBS"

    cell_detection_csv_path = os.path.join(non_error_dir, "cells", "cell_detection.csv")

    cell_det_df = pd.read_csv(cell_detection_csv_path, header=None, index_col=0)

    num_cells_detected = int(cell_det_df.loc["num_cells_detected", 1])
    num_focus_regions_scanned = int(cell_det_df.loc["num_focus_regions_scanned", 1])

    nonerror_df_dict["result_dir_name"].append(os.path.basename(non_error_dir))
    nonerror_df_dict["num_cells"].append(num_cells_detected)
    nonerror_df_dict["num_regions"].append(num_focus_regions_scanned)
    nonerror_df_dict["specimen_type"].append(specimen_type)

    for cell_name in cellnames:
        cellclass_path = os.path.join(non_error_dir, "cells", cell_name)

        # if the directory for the cell class exists
        if os.path.exists(cellclass_path):
            cellclass_dir = os.listdir(cellclass_path)
            # only keep the jpg files
            cellclass_jpgs = [f for f in cellclass_dir if f.endswith(".jpg")]

            nonerror_df_dict[str("num" + cell_name)].append(len(cellclass_jpgs))
        else:
            nonerror_df_dict[str("num" + cell_name)].append(0)

nonerror_df = pd.DataFrame(nonerror_df_dict)

# open the diff_data.csv file
diff_data = pd.read_csv(diff_data_path)

# only keeps the rows in nonerror_df that has a result_dir_name that is in diff_data's result_dir_name column
nonerror_df = nonerror_df[
    nonerror_df["result_dir_name"].isin(diff_data["result_dir_name"])
]

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Gathering data for the comparable differential groups
########################################################################################################################
########################################################################################################################
########################################################################################################################

for cellname in cellnames:
    # rename the column from numcellname to cellname
    nonerror_df = nonerror_df.rename(columns={f"num{cellname}": cellname})

# first remove all the columns named numX where X in removed_classes
for removed_class in removed_classes:
    nonerror_df = nonerror_df.drop(columns=[f"{removed_class}"])

for omitted_class in omitted_classes:
    nonerror_df = nonerror_df.drop(columns=[f"{omitted_class}"])

# renamed the num_cells column to num_objects
nonerror_df = nonerror_df.rename(columns={"num_cells": "num_objects"})

# create a new column named total_cells that is the sum of all the cell classes that are not in removed_classes
nonerror_df["num_cells"] = nonerror_df[non_removed_classes].sum(axis=1)

for differential_group in differential_group_dict:
    # get the list of cell classes in the differential group
    cell_classes = differential_group_dict[differential_group]

    # get the sum of the cell classes in the differential group
    nonerror_df[differential_group] = (
        nonerror_df[cell_classes].sum(axis=1) / nonerror_df["num_cells"]
    )

# now remove the columns in nonremoved_classes
nonerror_df = nonerror_df.drop(columns=non_removed_classes)

# print how many rows are in the nonerror_df
print(f"Number of rows in nonerror_df: {len(nonerror_df)}")

# save the non-error dataframe to a csv file at /media/hdd3/neo/pipeline_nonerror_aggregate_df.csv
nonerror_df.to_csv(
    "/media/hdd3/neo/DiffTransformerV1DataMini/pipeline_diff.csv", index=False
)
