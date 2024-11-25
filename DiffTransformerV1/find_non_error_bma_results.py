import os
import shutil
from tqdm import tqdm

results_folder = "/media/hdd3/neo/results_dir"

save_dir = "/media/hdd3/neo/DiffTransformerV1DataMini"
os.makedirs(save_dir, exist_ok=True)

removed_classes = ["U1", "PL2", "PL3", "ER5", "ER6", "U4"]

# get all the result directories
result_dirs = os.listdir(results_folder)

results_dirs_paths = [
    os.path.join(results_folder, result_dir)
    for result_dir in result_dirs
    if os.path.isdir(os.path.join(results_folder, result_dir))
]

bma_results_dirs_paths = [
    result_dir for result_dir in results_dirs_paths if "BMA-diff" in result_dir
]

num_error, num_non_error = 0, 0

non_error_dirs = []

for bma_results_dir in tqdm(bma_results_dirs_paths, desc="Checking directories"):
    # if an error.txt file exists in the directory, then it is an error directory
    if os.path.exists(os.path.join(bma_results_dir, "error.txt")):
        num_error += 1
    else:
        # if the cells subdirectory does not exist, then it is an error directory
        if not os.path.exists(os.path.join(bma_results_dir, "cells")):
            num_error += 1
        else:
            # get all the jpg files in the cells subdirectory, in a recursive fashion
            cells_dir = os.path.join(bma_results_dir, "cells")

            jpg_files = [
                os.path.join(root, file)
                for root, dirs, files in os.walk(cells_dir)
                for file in files
                if file.endswith(".jpg")
            ]

            final_jpg_files = []

            for jpg_file in jpg_files:
                # get the name of the directoy that the jpg file is in
                jpg_dir = os.path.dirname(jpg_file)

                # if the directory name is "blurry" or "cells" or any of the removed classes, then it is an error directory
                if os.path.basename(jpg_dir) in removed_classes:
                    continue
                elif os.path.basename(jpg_dir) == "blurry":
                    continue
                elif os.path.basename(jpg_dir) == "cells":
                    continue

                final_jpg_files.append(jpg_file)

            if len(final_jpg_files) < 100:
                num_error += 1
            else:
                num_non_error += 1
                non_error_dirs.append(bma_results_dir)


print(f"Found {num_error} error directories and {num_non_error} non-error directories")

for non_error_dir in tqdm(non_error_dirs, desc="Copying non-error directories"):
    save_path = os.path.join(save_dir, os.path.basename(non_error_dir))
    shutil.copytree(non_error_dir, save_path)
