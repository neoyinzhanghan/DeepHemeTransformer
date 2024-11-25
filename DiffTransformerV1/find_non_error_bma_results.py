import os

results_folder = "/media/hdd3/neo/results_dir"

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
for bma_results_dir in bma_results_dirs_paths:
    # if an error.txt file exists in the directory, then it is an error directory
    if os.path.exists(os.path.join(bma_results_dir, "error.txt")):
        num_error += 1
    else:
        num_non_error += 1

print(f"Found {num_error} error directories and {num_non_error} non-error directories")
