import os
from tqdm import tqdm

result_dir_path = "/media/hdd4/neo/results_dir"


# find the subdirs in the result_dir_path that does not start with ERROR_

subdirs = os.listdir(result_dir_path)
good_subdirs = []


for subdir in tqdm(subdirs):
    if not subdir.startswith("ERROR_"):
        good_subdirs.append(subdir)

# print the number of subdirs found
print(f"Number of subdirs found: {len(subdirs)}")
print(f"Number of good subdirs found: {len(good_subdirs)}")