import os
import torch
from tqdm import tqdm
from BMAassumptions import (
    non_removed_classes,
)

results_dir = "/media/hdd3/neo/DiffTransformerV1DataMini"
feature_name = "features_v3"
save_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"

os.makedirs(save_dir, exist_ok=True)

# get all the subdirectories in the results directory
all_subdirs = [
    d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
]


def get_stacked_feature_tensor(subdir, feature_name):
    """Get the stacked feature tensor from the given subdirectory."""
    list_of_feature_tensors = []
    list_of_feature_paths = []

    for cell_class in non_removed_classes:
        feature_dir_path = os.path.join(
            results_dir, subdir, cell_class, f"{feature_name}"
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

    stacked_feature_tensor = torch.stack(list_of_feature_tensors)

    return stacked_feature_tensor


for subdir in tqdm(all_subdirs, desc="Extracting features:"):
    stacked_feature_tensor = get_stacked_feature_tensor(subdir, feature_name)

    save_path = os.path.join(save_dir, f"{subdir}.pt")
    torch.save(stacked_feature_tensor, save_path)

print("Done extracting features.")
