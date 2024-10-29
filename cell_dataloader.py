import os
import h5py
import torch
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

    # Stack features and logits in batch dimension
    features = torch.stack(features)
    logits = torch.stack(logits)
    diff_tensor = torch.stack(diff_tensor)

    return features, logits, diff_tensor


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


class ImagePathDataset(Dataset):
    def __init__(self, image_paths):
        """
        Args:
            image_paths (list of str): List of paths to image files.
        """
        self.image_paths = image_paths
        self.current_idx = 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format

        idx = self.current_idx
        self.current_idx += 1

        return idx, image, img_path


class CellFeaturesLogitsDataset(Dataset):
    def __init__(self, metadata):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing features and logits.
            features_path (str): Path to the features file.
        """
        self.metadata = metadata

        # get the features_path column as a list
        self.features_path = metadata["features_path"].tolist()

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
        dataset = CellFeaturesLogitsDataset(metadata)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
