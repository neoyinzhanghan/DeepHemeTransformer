import os
import random
import pandas as pd
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm

differential_group_dict = {
    "blasts and blast-equivalents": ["M1"],
    "promyelocytes": ["M2"],
    "myelocytes": ["M3"],
    "metamyelocytes": ["M4"],
    "neutrophils/bands": ["M5", "M6"],
    "monocytes": ["MO2"],
    "eosinophils": ["E1", "E4"],
    "erythroid precursors": ["ER1", "ER2", "ER3", "ER4"],
    "lymphocytes": ["L2"],
    "plasma cells": ["L4"],
}

# grouped_label_to_index
grouped_label_to_index = {
    "blasts and blast-equivalents": 0,
    "promyelocytes": 1,
    "myelocytes": 2,
    "metamyelocytes": 3,
    "neutrophils/bands": 4,
    "monocytes": 5,
    "eosinophils": 6,
    "erythroid precursors": 7,
    "lymphocytes": 8,
    "plasma cells": 9,
    "skippocytes": 10,
}

# grouped_labels
grouped_labels = list(grouped_label_to_index.keys())


def create_plasma_cell_dataset_metadata(base_data_csv, plasma_cell_data_dir):
    """The base_metadata_csv should have columns idx,original_path,split
    The original_path should be the format '/media/ssd2/dh_labelled_data/DeepHeme1/UCSF_repo/ER2/45823.png'
    The immediate directory name should be the label
    """
    base_metadata = pd.read_csv(base_data_csv)
    # remove the split column
    base_metadata = base_metadata.drop(columns=["split"])

    # remove the idx column
    base_metadata = base_metadata.drop(columns=["idx"])

    # create the grouped_label column with mapping from differential_group_dict
    def get_mapped_label(cell_type):
        for group, cell_types in differential_group_dict.items():
            if cell_type in cell_types:
                return group
        return "skippocytes"

    base_metadata["grouped_label"] = base_metadata["original_path"].apply(
        lambda x: get_mapped_label(x.split("/")[-2])
    )

    # created the label column that is just the name of the dir
    base_metadata["label"] = base_metadata["original_path"].apply(
        lambda x: x.split("/")[-2]
    )

    # create a data_group column valued "original"
    base_metadata["data_group"] = "original"

    plasma_cell_jpgs = []
    plasma_cell_labels = []
    plasma_cell_grouped_labels = []
    plasma_cell_data_group = []

    # recursively find all the jpg files in the plasma_cell_data_dir
    print("Finding plasma cell images...")
    # First compute total number of directories for progress bar
    total = 0
    for root, dirs, files in os.walk(plasma_cell_data_dir):
        total += 1

    for root, dirs, files in tqdm(
        os.walk(plasma_cell_data_dir),
        desc="Finding plasma cell images",
        total=total,
    ):
        for file in tqdm(files, desc="Processing files in directory", total=len(files)):
            if file.endswith(".jpg"):
                plasma_cell_jpgs.append(os.path.join(root, file))
                plasma_cell_grouped_labels.append("plasma cells")
                plasma_cell_labels.append("L4")
                plasma_cell_data_group.append(
                    "plasma_cells_from_high_plasma_cell_slides"
                )

    print("Creating plasma cell metadata...")
    # add the plasma_cell_jpgs to the dataframe's original_path, label should be "L4"
    plasma_cell_metadata = pd.DataFrame(
        {
            "original_path": plasma_cell_jpgs,
            "grouped_label": plasma_cell_grouped_labels,
            "label": plasma_cell_labels,
            "data_group": plasma_cell_data_group,
        }
    )

    # concatenate the base_metadata and plasma_cell_metadata
    combined_metadata = pd.concat([base_metadata, plasma_cell_metadata])

    print("Assigning train/val/test splits...")
    # now randomly assign train, val, test split based on a 0.7, 0.15, 0.15 proportion
    # Create a new column for the split
    combined_metadata["split"] = np.random.choice(
        ["train", "val", "test"], size=len(combined_metadata), p=[0.7, 0.15, 0.15]
    )

    # save the combined_metadata to a csv
    combined_metadata.to_csv("combined_metadata.csv", index=False)

    return combined_metadata


class CustomDataset(Dataset):
    def __init__(
        self,
        base_data_dir,
        results_dirs_list,
        cell_types_list,
        base_data_sample_probability,
        sample_probabilities,
        transform=None,
    ):
        self.base_data_dir = base_data_dir
        self.results_dirs_list = results_dirs_list
        self.cell_types_list = cell_types_list
        self.base_data_sample_probability = base_data_sample_probability
        self.sample_probabilities = sample_probabilities
        self.transform = transform

        # Load the base dataset
        self.base_dataset = ImageFolder(base_data_dir)
        self.base_class_indices = {
            cls: [] for cls in range(len(self.base_dataset.classes))
        }
        for idx, (_, label) in enumerate(self.base_dataset.samples):
            self.base_class_indices[label].append(idx)

        self.num_base_data = len(self.base_dataset)
        self.num_data_points = 2 * self.num_base_data

    def sample_from_base_dir(self):
        # Class balancing: Choose a random class
        class_choice = random.choice(list(self.base_class_indices.keys()))
        # Randomly sample from that class
        sample_idx = random.choice(self.base_class_indices[class_choice])
        image, label = self.base_dataset.samples[sample_idx]
        image = Image.open(image)
        return image, label

    def sample_from_result_dirs(self):
        # Choose which result directory to sample from
        result_dir_choice = random.choices(
            self.results_dirs_list, self.sample_probabilities
        )[0]
        corresponding_cell_type = self.cell_types_list[
            self.results_dirs_list.index(result_dir_choice)
        ]

        # Sample a subdirectory
        sub_dirs = [
            d
            for d in os.listdir(result_dir_choice)
            if os.path.isdir(os.path.join(result_dir_choice, d))
        ]
        sub_dir_choice = random.choice(sub_dirs)

        # Sample a jpg file from the corresponding cell type folder
        cell_dir = os.path.join(
            result_dir_choice, sub_dir_choice, "cells", corresponding_cell_type
        )
        if os.path.exists(cell_dir):
            jpg_files = [f for f in os.listdir(cell_dir) if f.endswith(".jpg")]
            if jpg_files:
                image_file = random.choice(jpg_files)
                image_path = os.path.join(cell_dir, image_file)
                image = Image.open(image_path)
                label = self.base_dataset.class_to_idx[
                    corresponding_cell_type
                ]  # Assuming the cell types map to base classes
                return image, label
        return None, None

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, index):
        # Decide whether to sample from base data or result directories
        if random.random() < self.base_data_sample_probability:
            image, label = self.sample_from_base_dir()
        else:
            image, label = None, None
            while image is None:
                image, label = self.sample_from_result_dirs()

        if self.transform:
            image = self.transform(image)

        return image, label


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

cell_name_probabilities = {
    "B1": 0.01,
    "B2": 0.01,
    "E1": 0.02,
    "E4": 0.02,
    "ER1": 0.02,
    "ER2": 0.02,
    "ER3": 0.08,
    "ER4": 0.08,
    "ER5": 0.02,
    "ER6": 0.02,
    "L2": 0.1,
    "L4": 0.1,
    "M1": 0.1,
    "M2": 0.02,
    "M3": 0.03,
    "M4": 0.06,
    "M5": 0.08,
    "M6": 0.04,
    "MO2": 0.02,
    "PL2": 0.02,
    "PL3": 0.01,
    "U1": 0.1,
    "U4": 0.02,
}  # Total sum: 1.0


class CustomPlasmaCellDataset(Dataset):
    def __init__(
        self,
        combined_metadata_csv_path,
        cell_name_sampling_probabilities=cell_name_probabilities,
        transform=None,
        split="train",
    ):
        self.split = split
        combined_metadata = pd.read_csv(combined_metadata_csv_path)
        self.combined_metadata = combined_metadata[combined_metadata["split"] == split]
        self.class_metadata = {
            cellname: self.combined_metadata[
                self.combined_metadata["label"] == cellname
            ]
            for cellname in cellnames
        }
        self.cell_name_sampling_probabilities = cell_name_sampling_probabilities
        self.transform = transform

    def __len__(self):
        return len(self.combined_metadata)

    def __getitem__(self, idx):
        # Randomly sample a cell name based on probabilities
        cell_name = np.random.choice(
            list(self.cell_name_sampling_probabilities.keys()),
            p=list(self.cell_name_sampling_probabilities.values()),
        )

        # Randomly sample a row from that cell type's metadata
        cell_metadata = self.class_metadata[cell_name].sample(n=1).iloc[0]

        # Get image path and load image
        img_path = cell_metadata["original_path"]
        image = Image.open(img_path).convert("RGB")

        # Convert to tensor
        image = transforms.ToTensor()(image)

        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)
        # Get label
        grouped_label = cell_metadata["grouped_label"]
        label_idx = grouped_label_to_index[grouped_label]

        return image, label_idx


if __name__ == "__main__":
    base_data_csv = "/home/greg/Documents/neo/DeepHemeRetrain/scripts/new_metadata.csv"
    plasma_cell_data_dir = "/media/hdd3/neo/PCM_cells_annotated"
    save_path = "/media/hdd3/neo"

    def create_metadata_csv(root_dir):
        # Lists to store metadata
        metadata = []
        idx = 0

        # Walk through train, test, val directories
        for split in ["train", "test", "val"]:
            split_path = os.path.join(root_dir, split)

            # Check if the split directory exists
            if not os.path.exists(split_path):
                print(f"Warning: {split} directory not found")
                continue

            # Walk through all files in the split directory
            for root, _, files in os.walk(split_path):
                for file in files:
                    # Get the full path
                    file_path = os.path.join(root, file)

                    # Add to metadata
                    metadata.append(
                        {"idx": idx, "original_path": file_path, "split": split}
                    )
                    idx += 1

        # Create DataFrame and save to CSV
        df = pd.DataFrame(metadata)
        df.to_csv("new_metadata.csv", index=False)
        print(f"Created metadata CSV with {len(df)} entries")

    # Use the function
    root_dir = "/media/hdd3/neo/pooled_deepheme_data"
    create_metadata_csv(root_dir)

    combined_metadata = create_plasma_cell_dataset_metadata(
        base_data_csv, plasma_cell_data_dir
    )

    # save the combined_metadata to a csv
    combined_metadata.to_csv(
        os.path.join(save_path, "new_plasma_cell_deepheme_training_metadata.csv"),
        index=False,
    )
