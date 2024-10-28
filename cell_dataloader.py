import os
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
from BMAassumptions import BMA_final_classes


# Custom collate function to handle PIL images and names
def custom_collate_fn(batch):
    # Batch is a list of tuples (PIL image, image_name)
    indices, pil_images, paths = zip(*batch)
    return list(indices), list(pil_images), list(paths)


def get_diff_tensor(metadata, idx):
    # get the row of the metadata file corresponding to the idx
    row = metadata.iloc[idx]

    # get the values of the columns name in BMA_final_classes, convert to float and divide by 100
    # if the entry is NA, nan, empty or not a number, replace with 0
    diff_tensor = (
        torch.tensor(
            [
                row[class_name] if row[class_name] == row[class_name] else 0
                for class_name in BMA_final_classes
            ]
        ).float()
        / 100
    )

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
