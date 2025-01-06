import os
import pandas as pd
from tqdm import tqdm
from BMAassumptions import (
    removed_classes,
    omitted_classes,
    non_removed_classes,
    BMA_final_classes,
    differential_group_dict,
    cellnames,
)


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

metadata_dir = "/media/hdd3/neo/high_plasma_cell_slides_diff_data.csv"
cells_dir = "/media/hdd3/neo/high_plasma_cell_slides_cells"

# open the high_plasma_cell_slides.csv file
selected_slide_data = pd.read_csv(metadata_dir)

# non_error_dirs is the result_dir column of the selected_slide_data as a list of strings
non_error_dirs = selected_slide_data['result_dir'].tolist()

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
    "/media/hdd3/neo/high_plasma_cell_slides_pipeline_diff.csv", index=False
)
