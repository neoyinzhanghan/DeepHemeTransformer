import os
import h5py
from tqdm import tqdm

h5_dir_path = ""

# get all the paths to all the h5 files
h5_paths = [
    os.path.join(h5_dir_path, f) for f in os.listdir(h5_dir_path) if f.endswith(".h5")
]


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

        # if only contains 0 and 21, then return False, else return True
        return (
            set(class_indices) != {0, 21}
            or set(class_indices) != {21, 0}
            or set(class_indices) != {0}
            or set(class_indices) != {21}
            or set(class_indices) != {}
        )


valid_h5_files = []
invalid_h5_files = []

for h5_path in tqdm(h5_paths, desc="Checking cell classes..."):
    if check_cell_classes(h5_path):
        print(f"Found incorrect class indices in: {h5_path}")
        valid_h5_files.append(os.path.basename(h5_path))
    else:
        print(f"Correct class indices in: {h5_path}")
        invalid_h5_files.append(os.path.basename(h5_path))

print(f"Found {len(valid_h5_files)} valid h5 files.")
print(f"Found {len(invalid_h5_files)} invalid h5 files.")
