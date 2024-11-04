import os
import h5py
import torch
import numpy as np
import pandas as pd
from cell_dataloader import ImagePathDataset, custom_collate_fn
from deepheme import (
    Myresnext50,
    model_predict_batch,
    extract_features_batch,
    model_create,
)
from BMAassumptions import HemeLabel_ckpt_path

metadata_path = "/media/hdd3/neo/test_diff_results.csv"
results_dir = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir"
save_dir = "/media/hdd3/neo/DeepHemeTransformerDataTest"  # TODO remove the Test after debugging is complete
os.makedirs(save_dir, exist_ok=True)
model = model_create(HemeLabel_ckpt_path)


def extract_h5_data(result_folder, save_path, model, note=""):

    num_workers = 32

    # first recursively get the path to all the jpg images in the subdirectories of result_folder/cells
    jpg_paths = []
    for root, dirs, files in os.walk(os.path.join(result_folder, "cells")):
        for file in files:
            if file.endswith(".jpg"):
                jpg_paths.append(os.path.join(root, file))

    # shuffle the jpg_paths
    np.random.shuffle(jpg_paths) # TODO remove this line after debugging is complete

    N = len(jpg_paths)

    all_features = np.zeros((N, 2048))
    all_class_probs = np.zeros((N, 23))

    # CREATE A LENGTH N ARRAY TO STORE STRINGS WHICH ARE PATHS TO IMAGES
    all_paths = np.empty(N, dtype=object)

    metadata = pd.read_csv(metadata_path)

    dataset = ImagePathDataset(jpg_paths)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )

    for i, (indices, images, paths) in enumerate(dataloader):
        probabilities = model_predict_batch(model, images)
        features_batch = extract_features_batch(model, images)

        for j in range(len(paths)):
            idx = indices[j]
            all_features[idx] = features_batch[j]
            all_class_probs[idx] = probabilities[j]
            all_paths[idx] = paths[j]

        all_paths = np.array(all_paths, dtype="S")

    with h5py.File(save_path, "w") as f:
        f.create_dataset("features", data=all_features)
        f.create_dataset("class_probs", data=all_class_probs)
        f.create_dataset("paths", data=all_paths)

        # note save the note
        f.attrs["note"] = note

    return all_features, all_class_probs, all_paths


if __name__ == "__main__":
    from tqdm import tqdm

    # traverse through rows of metadata file
    metadata = pd.read_csv(metadata_path)

    metadata_dict = {
        "wsi_name": [],
        "result_dir_name": [],
        "features_path": [],
    }

    for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
        wsi_name = row["wsi_name"]
        result_dir_path = row["result_dir_path"]
        result_dir_name = os.path.basename(result_dir_path)
        actual_result_dir = os.path.join(results_dir, result_dir_name)

        features_path = os.path.join(save_dir, wsi_name.replace(".ndpi", ".h5"))

        # if the features file already exists, delete it
        if os.path.exists(features_path):
            os.remove(features_path)

        extract_h5_data(
            actual_result_dir,
            features_path,
            model,
            note=f"Extracted features from {result_dir_name}. Model: {model}. Using LLBMA pipeline before region classification update. 2024-10-27",
        )

        metadata_dict["wsi_name"].append(wsi_name)
        metadata_dict["result_dir_name"].append(result_dir_name)
        metadata_dict["features_path"].append(features_path)

    metadata_df = pd.DataFrame(metadata_dict)
    metadata_df.to_csv(os.path.join(save_dir, "features_metadata.csv"), index=False)
