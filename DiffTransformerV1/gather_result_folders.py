import os
import pandas as pd
from tqdm import tqdm

metadata_file_path = (
    "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
)
results_dir = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir"

save_dir = "/media/hdd3/neo/DiffTransformerV1Data"
os.makedirs(save_dir, exist_ok=True)

# open the metadata file
metadata = pd.read_csv(metadata_file_path)

# get the column result_dir_name as a list of strings
result_dir_names = metadata["result_dir_name"].tolist()

print(f"Found {len(result_dir_names)} result directories")

for result_dir_name in tqdm(result_dir_names, desc="Copying result directories"):
    result_dir = os.path.join(results_dir, result_dir_name)
    save_path = os.path.join(save_dir, result_dir_name)
    # use symbolic link to save space
    os.symlink(result_dir, save_path)
