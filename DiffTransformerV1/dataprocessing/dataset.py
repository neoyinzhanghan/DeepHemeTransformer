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
    """

    def __init__(self, feature_stacks_dir, diff_data_path):
        super().__init__()

        self.feature_stacks_dir = feature_stacks_dir
        self.diff_data = pd.read_csv(diff_data_path)
        self.result_dir_names = self.diff_data["result_dir_name"].tolist()

    def __len__(self):
        return len(self.result_dir_names)

    def __getitem__(self, idx):
        result_dir_name = self.result_dir_names[idx]
        feature_stack_path = os.path.join(
            self.feature_stacks_dir, f"{result_dir_name}.pt"
        )

        feature_stack = torch.load(feature_stack_path)  # this has shape [N, d]

        # randomly sample 100

        diff_list = []

        for cell_class in BMA_final_classes:
            diff_list.append(self.diff_data.loc[idx, cell_class])

        # if the feature stack is a numpy array, convert it to a tensor
        if isinstance(feature_stack, np.ndarray):
            feature_stack = torch.from_numpy(feature_stack)

        diff_tensor = torch.tensor(diff_list)

        return feature_stack, diff_tensor


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
