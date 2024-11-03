import os
import pandas as pd
from tqdm import tqdm

data_metadata_path = (
    "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
)
train_proportion = 0.8
val_proportion = 0.1
test_proportion = 0.1

# first open the metadata file
metadata = pd.read_csv(data_metadata_path)

# if the split column is present, remove it
if "split" in metadata.columns:
    metadata.drop("split", axis=1, inplace=True)

# get the number of rows in the metadata file
N = len(metadata)

# get the number of rows for the training, validation, and test sets
train_N = int(train_proportion * N)
val_N = int(val_proportion * N)
test_N = N - train_N - val_N

# shuffle the metadata rows
metadata = metadata.sample(frac=1).reset_index(drop=True)

# create the split column
metadata["split"] = ""

# assign the training, validation, and test set labels
metadata.loc[:train_N, "split"] = "train"
metadata.loc[train_N : train_N + val_N, "split"] = "val"
metadata.loc[train_N + val_N :, "split"] = "test"
