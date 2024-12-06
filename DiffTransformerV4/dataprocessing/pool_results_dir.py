import os
from tqdm import tqdm

tmp_batched_results_dir = "/media/hdd3/neo/glv3_results_dir_batched"
save_dir = "/media/hdd3/neo/DeepHemeTransformerData3000"

os.makedirs(save_dir, exist_ok=True)

# find all the subdirectories in the tmp_batched_results_dir
subdirs = [
    os.path.join(tmp_batched_results_dir, d)
    for d in os.listdir(tmp_batched_results_dir)
    if os.path.isdir(os.path.join(tmp_batched_results_dir, d))
]

all_subsubdirs = []

for subdir in subdirs:
    subsubdirs = [
        os.path.join(subdir, d)
        for d in os.listdir(subdir)
        if os.path.isdir(os.path.join(subdir, d))
    ]
    all_subsubdirs.extend(subsubdirs)

# in the save_dir make a symbolic link to each of the subsubdirs
for subsubdir in tqdm(all_subsubdirs, desc="Creating symbolic links"):
    subsubdir_name = subsubdir.split("/")[-1]
    save_subsubdir = os.path.join(save_dir, subsubdir_name)
    os.symlink(subsubdir, save_subsubdir)
