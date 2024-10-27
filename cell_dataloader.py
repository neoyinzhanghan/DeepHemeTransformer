import os
import torch
from torch.utils.data import Dataset
from PIL import Image

# Custom collate function to handle PIL images and names
def custom_collate_fn(batch):
    # Batch is a list of tuples (PIL image, image_name)
    indices, pil_images, paths = zip(*batch)
    return list(indices), list(pil_images), list(paths)

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
        image = Image.open(img_path).convert('RGB')  # Ensure RGB format

        idx = self.current_idx
        self.current_idx += 1

        return idx, image, img_path
