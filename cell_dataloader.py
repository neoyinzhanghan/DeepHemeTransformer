import os
import h5py
import torch
import random
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from PIL import Image
from BMAassumptions import BMA_final_classes


# Custom collate function to handle features, logits, diff_tensor
def custom_collate_fn(batch):
    # Batch is a list of tuples features, logits, diff_tensor
    features, logits, diff_tensor = zip(*batch)

    return list(features), list(logits), list(diff_tensor)


def clean_diff_value(value):
    # if the entry is NA, nan, empty, non-numeric string, replace with 0
    if (
        value != value
        or value == "NA"
        or value == ""
        or not str(value).replace(".", "", 1).isdigit()
    ):
        return 0
    else:
        return float(value)


def get_diff_tensor(metadata, idx):
    # get the row of the metadata file corresponding to the idx
    row = metadata.iloc[idx]

    diff_list = [row[class_name] for class_name in BMA_final_classes]
    diff_tensor = torch.tensor([clean_diff_value(diff) for diff in diff_list]).float()

    return diff_tensor


# class ImagePathDataset(Dataset):
#     def __init__(self, image_paths):
#         """
#         Args:
#             image_paths (list of str): List of paths to image files.
#         """
#         self.image_paths = image_paths
#         self.current_idx = 0

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_path = self.image_paths[idx]

#         # Load image
#         image = Image.open(img_path).convert("RGB")  # Ensure RGB format

#         idx = self.current_idx
#         self.current_idx += 1

#         return idx, image, img_path


class ImagePathDataset(Dataset):
    def __init__(self, image_paths):
        """
        Args:
            image_paths (list of str): List of paths to image files.
        """
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format

        return idx, image, img_path


class CellFeaturesLogitsDataset(Dataset):
    def __init__(self, metadata, split="train"):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing features and logits.
            features_path (str): Path to the features file.
            split (str): Dataset split (train, val, test).
        """

        # get the features_path column as a list for all the rows where split is equal to the split
        self.metadata = metadata[metadata["split"] == split]
        self.features_path = self.metadata["features_path"].tolist()

        self.split = split
        assert self.split in ["train", "val", "test"]

        # shuffle the features_path list
        random.shuffle(self.features_path)

        # # randomly sample 10% of the data
        # self.features_path = random.sample(
        #     self.features_path, int(0.25 * len(self.features_path))
        # )

    def __len__(self):
        return len(self.features_path)

    def __getitem__(self, idx):
        feature_path = self.features_path[idx]

        diff_tensor = get_diff_tensor(self.metadata, idx)

        # f.create_dataset('features', data=all_features)
        # f.create_dataset('class_probs', data=all_class_probs)

        with h5py.File(feature_path, "r") as f:
            features = f["features"][:]
            logits = f["class_probs"][:]

            # convert to tensor
            features = torch.tensor(features).float()
            logits = torch.tensor(logits).float()

        return features, logits, diff_tensor


class CellFeaturesDataModule(pl.LightningDataModule):
    def __init__(self, metadata_file: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load metadata
        metadata = pd.read_csv(self.metadata_file)

        # Determine dataset lengths based on the split ratios
        train_dataset = CellFeaturesLogitsDataset(metadata, split="train")
        val_dataset = CellFeaturesLogitsDataset(metadata, split="val")
        test_dataset = CellFeaturesLogitsDataset(metadata, split="test")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )
