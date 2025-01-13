import os
import pandas as pd
from tqdm import tqdm

result_dir_path = "/media/hdd4/neo/results_dir"
service_run_history_path = "/media/hdd2/neo/high_plasma_cell_processing_metadata.csv"
diff_data_path = "/media/hdd3/neo/diff_data_processed.csv"

# open the diff_data_processed.csv file
diff_data = pd.read_csv(diff_data_path)

# replace na values with 0
diff_data.fillna(0, inplace=True)

threshold = 50
# find the rows where the column plasma cells is >= 75
high_plasma_cell_slides = diff_data[diff_data["plasma cells"] >= threshold]

# print the number of rows where the column plasma cells is >= 75
print(f"Total number of rows: {len(diff_data)}")
print(
    f"Number of rows where plasma cells >= {threshold}: {len(high_plasma_cell_slides)}"
)

# save the high_plasma_cell_slides to a csv file at /media/hdd3/neo/high_plasma_cell_slides.csv
high_plasma_cell_slides.to_csv(
    "/media/hdd2/neo/high_plasma_cell_slides.csv", index=False
)

# open the pipeline run history csv file
pipeline_run_history = pd.read_csv(service_run_history_path)

df_dict = {
    "accession_number": [],
    "wsi_name": [],
    "part_description": [],
    "result_dir": [],
    "blasts": [],
    "blast-equivalents": [],
    "promyelocytes": [],
    "myelocytes": [],
    "metamyelocytes": [],
    "neutrophils/bands": [],
    "monocytes": [],
    "eosinophils": [],
    "erythroid precursors": [],
    "lymphocytes": [],
    "plasma cells": [],
}

# get the data type of the result_dir_name column
result_dir_name_dtype = pipeline_run_history["result_dir_name"].dtype
print(f"Data type of result_dir_name column: {result_dir_name_dtype}")

# make sure the result_dir_name column is a string
pipeline_run_history["result_dir_name"] = pipeline_run_history[
    "result_dir_name"
].astype(str)

# only keep the rows in pipeline_run_history where the result_dir_name is not empty and not start with "ERROR"
pipeline_run_history = pipeline_run_history[
    (pipeline_run_history["result_dir_name"] != "")
    & (~pipeline_run_history["result_dir_name"].str.startswith("ERROR"))
]

# print the number of rows in pipeline_run_history
print(f"Number of rows in pipeline_run_history: {len(pipeline_run_history)}")

for idx, row in tqdm(
    pipeline_run_history.iterrows(),
    total=len(pipeline_run_history),
    desc="Processing Slides",
):
    wsi_name = row["wsi_name"]
    accession_number = wsi_name.split(";")[0]

    result_dir_name = row["result_dir_name"]
    result_dir_path = os.path.join(
        "/media/hdd2/neo/HighPlasmaCellLLBMAResults", result_dir_name
    )
    if accession_number in high_plasma_cell_slides["specnum_formatted"].values:
        diff_data_row = high_plasma_cell_slides[
            high_plasma_cell_slides["specnum_formatted"] == accession_number
        ]

        df_dict["accession_number"].append(accession_number)
        df_dict["wsi_name"].append(wsi_name)
        df_dict["part_description"].append(diff_data_row["part_description"].values[0])
        df_dict["result_dir"].append(result_dir_path)
        df_dict["blasts"].append(diff_data_row["blasts"].values[0])
        df_dict["blast-equivalents"].append(
            diff_data_row["blast-equivalents"].values[0]
        )
        df_dict["promyelocytes"].append(diff_data_row["promyelocytes"].values[0])
        df_dict["myelocytes"].append(diff_data_row["myelocytes"].values[0])
        df_dict["metamyelocytes"].append(diff_data_row["metamyelocytes"].values[0])
        df_dict["neutrophils/bands"].append(
            diff_data_row["neutrophils/bands"].values[0]
        )
        df_dict["monocytes"].append(diff_data_row["monocytes"].values[0])
        df_dict["eosinophils"].append(diff_data_row["eosinophils"].values[0])
        df_dict["erythroid precursors"].append(
            diff_data_row["erythroid precursors"].values[0]
        )
        df_dict["lymphocytes"].append(diff_data_row["lymphocytes"].values[0])
        df_dict["plasma cells"].append(diff_data_row["plasma cells"].values[0])

high_plasma_cell_slides_data = pd.DataFrame(df_dict)

# sum the blasts and blast-equivalents columns together into a new column called "blasts and blast-equivalents"
high_plasma_cell_slides_data["blasts and blast-equivalents"] = (
    high_plasma_cell_slides_data["blasts"]
    + high_plasma_cell_slides_data["blast-equivalents"]
)

# now remove the blasts and blast-equivalents columns
high_plasma_cell_slides_data.drop(columns=["blasts", "blast-equivalents"], inplace=True)

print(f"Found {len(high_plasma_cell_slides_data)} slides with high plasma cells")

# save the high_plasma_cell_slides_data to a csv file at /media/hdd3/neo/high_plasma_cell_slides_diff_data.csv
high_plasma_cell_slides_data.to_csv(
    "/media/hdd2/neo/high_plasma_cell_slides_diff_data.csv", index=False
)
