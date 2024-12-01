import pandas as pd

num_train = 20
diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv"
diff_data = pd.read_csv(diff_data_path)

# shuffle the data
diff_data = diff_data.sample(frac=1).reset_index(drop=True)

# add a split column
diff_data["split"] = "train"

# only keep the first num_train rows in the training set
diff_data.loc[num_train:, "split"] = "val"

# save the split data
split_diff_data_path = (
    "/media/hdd3/neo/DiffTransformerV1DataMini/subsampled_split_diff_data.csv"
)

diff_data.to_csv(split_diff_data_path, index=False)
