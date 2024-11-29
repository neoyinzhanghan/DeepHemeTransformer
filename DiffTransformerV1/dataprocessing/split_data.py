import pandas as pd

diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv"
diff_data = pd.read_csv(diff_data_path)

train_prop, val_prop = 0.8, 0.2
assert round(train_prop + val_prop, 3) == 1.0, "Train and val proportions must sum to 1"

num_train = int(train_prop * len(diff_data))
num_val = int(val_prop * len(diff_data))

# shuffle the data
diff_data = diff_data.sample(frac=1).reset_index(drop=True)

# add a split column
diff_data["split"] = "train"
diff_data.loc[num_train : num_train + num_val, "split"] = "val"

# save the split data
split_diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"
diff_data.to_csv(split_diff_data_path, index=False)

print(f"Saved split data to {split_diff_data_path}")
