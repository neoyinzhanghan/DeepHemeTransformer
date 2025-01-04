import os

result_dir_path = "/media/hdd4/neo/results_dir"


# find the subdirs in the result_dir_path that does not start with ERROR_

subdirs = [f for f in os.listdir(result_dir_path) if os.path.isdir(os.path.join(result_dir_path, f)) and not f.startswith("ERROR_")]

# print the number of subdirs found
print(f"Number of subdirs found: {len(subdirs)}")