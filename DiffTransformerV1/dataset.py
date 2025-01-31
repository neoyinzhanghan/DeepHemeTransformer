import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from BMAassumptions import BMA_final_classes


class TensorStackDataset(Dataset):
    """Dataset class for loading stacked feature tensors from a directory.

    feature_stacks_dir: str
    diff_data_path: str
    split: str (train, val, test)
    """

    def __init__(
        self, feature_stacks_dir, diff_data_path, split="train", num_cells=100
    ):
        super().__init__()

        self.feature_stacks_dir = feature_stacks_dir
        self.diff_data = pd.read_csv(diff_data_path)
        self.split = split
        self.num_cells = num_cells
        self.diff_data = self.diff_data[self.diff_data["split"] == self.split]

        # make sure to reset the index after filtering
        self.diff_data = self.diff_data.reset_index(drop=True)

        # print the number of rows in the differential data
        print(f"Number of rows in differential data: {len(self.diff_data)}")
        # print the largest index in the differential data
        print(f"Largest index in differential data: {self.diff_data.index.max()}")

        self.result_dir_names = self.diff_data["result_dir_name"].tolist()

    def __len__(self):
        return len(self.result_dir_names)

    def __getitem__(self, idx):
        result_dir_name = self.result_dir_names[idx]
        feature_stack_path = os.path.join(
            self.feature_stacks_dir, f"{result_dir_name}.pt"
        )

        feature_stack = torch.load(feature_stack_path)  # this has shape [N, d]

        # take the first num_cells samples without any shuffling or random sampling
        # assert (
        #     feature_stack.shape[0] > self.num_cells
        # ), "Feature stack has fewer than num_cells samples"
        # if feature_stack.shape[0] > self.num_cells:
        #     feature_stack = feature_stack[: self.num_cells]

        # randomly sample num_cells to get shape [num_cells, d]

        if feature_stack.shape[0] > self.num_cells:
            idxs = np.random.choice(
                feature_stack.shape[0], self.num_cells, replace=False
            )
            feature_stack = feature_stack[idxs]

        else:  # bootstrap if less than num_cells
            idxs = np.random.choice(
                feature_stack.shape[0], self.num_cells, replace=True
            )
            feature_stack = feature_stack[idxs]

        diff_list = []

        for cell_class in BMA_final_classes:
            diff_list.append(self.diff_data.loc[idx, cell_class])

        # if the feature stack is a numpy array, convert it to a tensor
        if isinstance(feature_stack, np.ndarray):
            feature_stack = torch.from_numpy(feature_stack)

        diff_tensor = torch.tensor(diff_list)

        return feature_stack, diff_tensor


from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class TensorStackDataModule(LightningDataModule):
    """
    DataModule for managing the TensorStackDataset with pre-split data.

    Args:
        feature_stacks_dir (str): Directory containing feature stack `.pt` files.
        diff_data_path (str): Path to the CSV file containing differential data.
        bma_final_classes (list): List of cell classes for the differential tensor.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
    """

    def __init__(
        self,
        feature_stacks_dir,
        diff_data_path,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.feature_stacks_dir = feature_stacks_dir
        self.diff_data_path = diff_data_path
        self.bma_final_classes = BMA_final_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Set up the datasets for train, validation, and test splits.
        """
        self.train_dataset = TensorStackDataset(
            feature_stacks_dir=self.feature_stacks_dir,
            diff_data_path=self.diff_data_path,
            split="train",
        )
        self.val_dataset = TensorStackDataset(
            feature_stacks_dir=self.feature_stacks_dir,
            diff_data_path=self.diff_data_path,
            split="val",
        )
        self.test_dataset = TensorStackDataset(
            feature_stacks_dir=self.feature_stacks_dir,
            diff_data_path=self.diff_data_path,
            split="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    # Example paths
    feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"  # Replace with your directory
    diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv"  # Replace with your CSV file

    # Instantiate the dataset
    dataset = TensorStackDataset(feature_stacks_dir, diff_data_path)

    # Test shapes of feature_stack and diff_tensor
    for idx in range(min(len(dataset), 5)):  # Test first 5 items or fewer
        feature_stack, diff_tensor = dataset[idx]
        print(f"Index {idx}:")
        print(f"Feature stack shape: {feature_stack.shape}")
        print(f"Diff tensor shape: {diff_tensor.shape}")
