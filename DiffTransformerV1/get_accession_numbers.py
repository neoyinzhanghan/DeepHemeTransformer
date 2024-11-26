import os
import pandas as pd
from tqdm import tqdm

good_results_dir = "/media/hdd3/neo/DiffTransformerV1DataMini"
pipeline_run_history_path = "/media/hdd3/neo/results_dir/pipeline_run_history.csv"
pipeline_run_history = pd.read_csv(pipeline_run_history_path)

# get a list of the result directories
result_dirs = os.listdir(good_results_dir)

metadata = {
    "result_dir_name": [],
    "accession_number": [],
    "wsi_name": [],
}


for result_dir in tqdm(result_dirs, desc="Getting accession numbers"):
    result_dir_path = os.path.join("/media/hdd3/neo/results_dir", result_dir)
    # look for the raow in pipeline_run_history where result_dir is in result_dir column
    row = pipeline_run_history[pipeline_run_history["result_dir"] == result_dir_path]

    assert len(row) == 1, f"Found {len(row)} rows for {result_dir}"

    row = row.iloc[0]

    wsi_name = row["wsi_name"]
    accession_number = wsi_name.split(";")[0]

    metadata["result_dir_name"].append(result_dir)
    metadata["accession_number"].append(accession_number)
    metadata["wsi_name"].append(wsi_name)

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(
    "/media/hdd3/neo/DiffTransformerV1DataMini/wsi_metadata.csv", index=False
)
