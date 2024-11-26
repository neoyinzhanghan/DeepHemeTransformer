import os
import torch
import pandas as pd
from tqdm import tqdm
from BMAassumptions import (
    non_removed_classes,
)

results_dir = "/media/hdd3/neo/DiffTransformerV1DataMini"
diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv"
feature_name = "features_v3"
save_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"

os.makedirs(save_dir, exist_ok=True)

# get the column named result_dir_name from the diff_data.csv file as a list of strings
diff_data = pd.read_csv(diff_data_path)
result_dir_names = diff_data["result_dir_name"].tolist()


def get_stacked_feature_tensor(subdir, feature_name):
    """Get the stacked feature tensor from the given subdirectory."""
    list_of_feature_tensors = []
    list_of_feature_paths = []

    for cell_class in non_removed_classes:
        feature_dir_path = os.path.join(
            results_dir, subdir, "cells", cell_class, f"{feature_name}"
        )

        if not os.path.exists(feature_dir_path):
            continue
        list_of_feature_paths.extend(
            [
                os.path.join(feature_dir_path, f)
                for f in os.listdir(feature_dir_path)
                if f.endswith(".pt")
            ]
        )

    for feature_path in list_of_feature_paths:
        feature_tensor = torch.load(feature_path)
        list_of_feature_tensors.append(feature_tensor)

    try:
        stacked_feature_tensor = torch.stack(list_of_feature_tensors)
    except RuntimeError as e:
        print(f"Error stacking feature tensors for {subdir}: {str(e)}")
        raise e

    return stacked_feature_tensor


for subdir in tqdm(result_dir_names, desc="Extracting features:"):
    stacked_feature_tensor = get_stacked_feature_tensor(subdir, feature_name)

    save_path = os.path.join(save_dir, f"{subdir}.pt")
    torch.save(stacked_feature_tensor, save_path)

print("Done extracting features.")
