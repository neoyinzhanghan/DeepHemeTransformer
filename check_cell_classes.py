import os
import h5py
import pandas as pd
from tqdm import tqdm

h5_dir_path = "/media/hdd3/neo/DeepHemeTransformerData"

# get all the paths to all the h5 files
h5_paths = [
    os.path.join(h5_dir_path, f) for f in os.listdir(h5_dir_path) if f.endswith(".h5")
]

validity_check_results = {
    "h5_name": [],
    "unique_classes": [],
    "num_samples": [],
}

class_tally = []


def check_cell_classes(h5_path):
    # firt open the h5 file
    with h5py.File(h5_path, "r") as f:
        # get the class_probs dataset
        class_probs = f["class_probs"][:]

        # class probs has shape (num_samples, num_classes)
        # take the argmax over th num_classes to get a tensor of shape (num_samples,) with the class indices as integers
        class_indices = class_probs.argmax(axis=1)

        # print the unique class indices
        print(f"Unique class indices: {set(class_indices)}")

        num_samples = class_probs.shape[0]

        # appen the class indices to the class_tally list
        class_tally.append(class_indices)

        # return false if class_indices is a subset of {0, 21}
        return not set(class_indices).issubset({0, 21}), class_indices, num_samples


valid_h5_files = []
invalid_h5_files = []

for h5_path in tqdm(h5_paths, desc="Checking cell classes..."):
    validity, class_indices, num_samples = check_cell_classes(h5_path)
    if validity:
        print(f"Found incorrect class indices in: {h5_path}")
        valid_h5_files.append(os.path.basename(h5_path))
    else:
        print(f"Correct class indices in: {h5_path}")
        invalid_h5_files.append(os.path.basename(h5_path))

    validity_check_results["h5_name"].append(os.path.basename(h5_path))
    validity_check_results["unique_classes"].append(set(class_indices))
    validity_check_results["num_samples"].append(num_samples)

# make a bar plot of the class tally
import matplotlib.pyplot as plt

plt.hist(class_tally, bins=22)
plt.xlabel("Class index")

# save the plot at the h5 directory
plt.savefig(os.path.join(h5_dir_path, "class_tally.png"))

print(f"Found {len(valid_h5_files)} valid h5 files.")
print(f"Found {len(invalid_h5_files)} invalid h5 files.")

validity_check_results_df = pd.DataFrame(validity_check_results)
validity_check_results_df.to_csv(
    os.path.join(h5_dir_path, "validity_check_results.csv"), index=False
)
