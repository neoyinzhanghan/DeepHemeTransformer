import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from BMAassumptions import BMA_final_classes
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class TensorStackDatasetV5(Dataset):
    """Dataset class for loading stacked feature tensors from a directory.

    feature_stacks_dir: str
    logit_stacks_dir: str
    diff_data_path: str
    split: str (train, val, test)
    dx_data_path: str
    """

    def __init__(
        self,
        feature_stacks_dir,
        logit_stacks_dir,
        diff_data_path,
        dx_data_path,
        split="train",
        min_num_cells=100,
        max_num_cells=3000,
        N=3000,
    ):
        super().__init__()

        self.feature_stacks_dir = feature_stacks_dir
        self.logit_stacks_dir = logit_stacks_dir
        self.diff_data = pd.read_csv(diff_data_path)
        self.dx_data = pd.read_csv(dx_data_path)
        self.split = split
        self.min_num_cells = min_num_cells
        self.max_num_cells = max_num_cells
        self.N = N
        self.diff_data = self.diff_data[self.diff_data["split"] == self.split]

        # only keep the diff_data rows that have a corresponding dx in the dx_data, matching using accession_number
        self.diff_data = self.diff_data[
            self.diff_data["accession_number"].isin(self.dx_data["accession_number"])
        ]

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
        accession_number = self.diff_data.loc[idx, "accession_number"]
        dx = self.dx_data[self.dx_data["accession_number"] == accession_number][
            "dx"
        ].values[0]

        if dx == "Normal BMA":
            label = 0
        elif dx == "Plasma cell myeloma":
            label = 1
        else:
            raise ValueError(f"Unsupported dx: {dx}")

        feature_stack_path = os.path.join(
            self.feature_stacks_dir, f"{result_dir_name}.pt"
        )
        logit_stack_path = os.path.join(self.logit_stacks_dir, f"{result_dir_name}.pt")
        logit_stack = torch.load(logit_stack_path)

        feature_stack = torch.load(feature_stack_path)  # this has shape [N, d]

        num_cells = np.random.randint(self.min_num_cells, self.max_num_cells)

        # num_cells = np.random.randint(self.min_num_cells, self.max_num_cells)

        if feature_stack.shape[0] > num_cells:
            idxs = np.random.choice(feature_stack.shape[0], num_cells, replace=False)
            feature_stack = feature_stack[idxs]
            logit_stack = logit_stack[idxs]

        # take the first num_cells samples without any shuffling or random sampling
        # assert (
        #     feature_stack.shape[0] > self.num_cells
        # ), "Feature stack has fewer than num_cells samples"
        # if feature_stack.shape[0] > self.num_cells:
        #     feature_stack = feature_stack[: self.num_cells]

        # randomly sample num_cells to get shape [num_cells, d]

        if feature_stack.shape[0] > self.N:
            idxs = np.random.choice(feature_stack.shape[0], self.N, replace=False)
            feature_stack = feature_stack[idxs]
            logit_stack = logit_stack[idxs]

            non_padding_mask = torch.ones(self.N)
            non_padding_mask[feature_stack.shape[0] :] = 0

        else:  # take all the data and padding with zeros
            feature_padding = np.zeros(
                (self.N - feature_stack.shape[0], feature_stack.shape[1])
            )
            logit_padding = np.zeros(
                (self.N - logit_stack.shape[0], logit_stack.shape[1])
            )

            non_padding_mask = torch.ones(self.N)
            non_padding_mask[feature_stack.shape[0] :] = 0

            feature_stack = np.concatenate((feature_stack, feature_padding), axis=0)
            logit_stack = np.concatenate((logit_stack, logit_padding), axis=0)

            feature_stack = torch.from_numpy(feature_stack)
            logit_stack = torch.from_numpy(logit_stack)

            # make sure that the feature stack is Double
        feature_stack = feature_stack.float()
        logit_stack = logit_stack.float()
        non_padding_mask = non_padding_mask.float()

        diff_list = []

        for cell_class in BMA_final_classes:
            diff_list.append(self.diff_data.loc[idx, cell_class])

        # if the feature stack is a numpy array, convert it to a tensor
        if isinstance(feature_stack, np.ndarray):
            feature_stack = torch.from_numpy(feature_stack)

        if isinstance(logit_stack, np.ndarray):
            logit_stack = torch.from_numpy(logit_stack)

        diff_tensor = torch.tensor(diff_list)

        return feature_stack, logit_stack, non_padding_mask, diff_tensor, label


class TensorStackDataModuleV5(LightningDataModule):
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
        logit_stacks_dir,
        diff_data_path,
        dx_data_path,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.feature_stacks_dir = feature_stacks_dir
        self.logit_stacks_dir = logit_stacks_dir
        self.diff_data_path = diff_data_path
        self.dx_data_path = dx_data_path
        self.bma_final_classes = BMA_final_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Set up the datasets for train, validation, and test splits.
        """
        self.train_dataset = TensorStackDatasetV5(
            feature_stacks_dir=self.feature_stacks_dir,
            logit_stacks_dir=self.logit_stacks_dir,
            diff_data_path=self.diff_data_path,
            dx_data_path=self.dx_data_path,
            split="train",
        )
        self.val_dataset = TensorStackDatasetV5(
            feature_stacks_dir=self.feature_stacks_dir,
            logit_stacks_dir=self.logit_stacks_dir,
            diff_data_path=self.diff_data_path,
            dx_data_path=self.dx_data_path,
            split="val",
        )
        self.test_dataset = TensorStackDatasetV5(
            feature_stacks_dir=self.feature_stacks_dir,
            logit_stacks_dir=self.logit_stacks_dir,
            diff_data_path=self.diff_data_path,
            dx_data_path=self.dx_data_path,
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
    logit_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"  # Replace with your directory
    diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"  # Replace with your CSV file
    dx_data_path = "/media/hdd3/neo/dx_data_test.csv"  # Replace with your CSV file

    # Instantiate the dataset
    dataset = TensorStackDatasetV5(
        feature_stacks_dir,
        logit_stacks_dir,
        diff_data_path,
        dx_data_path=dx_data_path,
        split="train",
    )

    # Test shapes of feature_stack and diff_tensor
    for idx in range(min(len(dataset), 5)):  # Test first 5 items or fewer
        feature_stack, logit_stack, non_padding_mask, diff_tensor, label = dataset[idx]
        print(f"Index {idx}:")
        print(f"Feature stack shape: {feature_stack.shape}")
        print(f"Logit stack shape: {logit_stack.shape}")
        print(f"Non-padding mask shape: {non_padding_mask.shape}")
        print(f"Diff tensor shape: {diff_tensor.shape}")
        print(f"dx label: {label}")

    # Instantiate the datamodule
    datamodule = TensorStackDataModuleV5(
        feature_stacks_dir,
        logit_stacks_dir,
        diff_data_path,
        dx_data_path,
        batch_size=32,
        num_workers=4,
    )

    # Setup the datamodule
    datamodule.setup()

    # get the first batch of the train dataloader
    for batch in datamodule.train_dataloader():
        feature_stack, logit_stack, non_padding_mask, diff_tensor, label = batch
        print(f"Feature stack shape: {feature_stack.shape}")
        print(f"Logit stack shape: {logit_stack.shape}")
        print(f"Non-padding mask shape: {non_padding_mask.shape}")
        print(f"Diff tensor shape: {diff_tensor.shape}")
        print(f"dx label: {label}")
        break
