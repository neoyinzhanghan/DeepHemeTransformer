import os
import ray
import torch
import shutil
import numpy as np
import pandas as pd
from LLBMA.resources.BMAassumptions import (
    cell_clf_batch_size,
    num_labellers,
    HemeLabel_ckpt_path,
)
from LLBMA.brain.labeller.HemeLabelLightningManager import (
    HemeLabelLightningManager,
    model_create,
    predict_batch,
)
from LLBMA.brain.utils import create_list_of_batches_from_list
from ray.exceptions import RayTaskError
from tqdm import tqdm
from tqdm import tqdm
from BMAassumptions import (
    non_removed_classes,
)
from PIL import Image

results_dir = "/media/hdd3/neo/DiffTransformerV1DataMini"
diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv"
feature_name = "features_v3"
save_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"


model = model_create(HemeLabel_ckpt_path)


def get_cell_path(feature_path):
    # first separate the file name from the dir path
    feature_path_split = feature_path.split("/")
    feature_file_name = feature_path_split[-1]
    # replace .pt with .jpg and we have the cell_image_name
    cell_image_name = feature_file_name.replace(".pt", ".jpg")

    cell_dir = feature_path_split[-3]

    cell_image_path = os.path.join(cell_dir, cell_image_name)

    return cell_image_path


# if the save directory already exists, delete it
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

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

    cell_paths = []
    cell_pil_images = []

    for feature_path in list_of_feature_paths:
        feature_tensor = torch.load(feature_path)

        cell_path = get_cell_path(feature_path)
        cell_paths.append(cell_path)

        cell_pil_image = Image.open(cell_path)
        cell_pil_images.append(cell_pil_image)

        # if the feature tensor is a numpy array, convert it to a tensor
        if isinstance(feature_tensor, np.ndarray):
            feature_tensor = torch.from_numpy(feature_tensor)

        list_of_feature_tensors.append(feature_tensor)

    try:
        stacked_feature_tensor = torch.stack(list_of_feature_tensors)
    except RuntimeError as e:
        print(f"Error stacking feature tensors for {subdir}: {str(e)}")
        raise e

    logits = predict_batch(cell_pil_images, model)

    print(type(logits))
    print(len(logits))
    print(type(logits[0]))
    print(logits[0].shape)

    import sys

    sys.exit()

    return stacked_feature_tensor


for subdir in tqdm(result_dir_names, desc="Extracting features:"):
    stacked_feature_tensor = get_stacked_feature_tensor(subdir, feature_name)

    save_path = os.path.join(save_dir, f"{subdir}.pt")
    torch.save(stacked_feature_tensor, save_path)

print("Done extracting features.")
