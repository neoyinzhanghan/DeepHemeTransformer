import os
import pandas as pd
from tqdm import tqdm
import shutil

selected_slide_data_path = "/media/hdd3/neo/high_plasma_cell_slides_diff_data.csv"
save_dir = "/media/hdd3/neo/high_plasma_cell_slides_cells"

os.makedirs(save_dir, exist_ok=True)

# open the high_plasma_cell_slides.csv file
selected_slide_data = pd.read_csv(selected_slide_data_path)

# get the result_dir column of the selected_slide_data as a list of strings
result_dirs = selected_slide_data["result_dir"].tolist()

for result_dir in tqdm(result_dirs, desc="Processing Slides"):
    result_dir_name = result_dir.split("/")[-1]
    cells_dir = os.path.join(result_dir, "cells")
    target_dir = os.path.join(save_dir, result_dir_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # recursively find the paths of all the jpg files in the cells_dir
    cells_files = []
    for root, dirs, files in os.walk(cells_dir):
        # Get the relative path from cells_dir
        rel_path = os.path.relpath(root, cells_dir)
        
        for file in files:
            if file.endswith(".jpg"):
                source_path = os.path.join(root, file)
                
                # Create the same subfolder structure in target_dir
                target_subdir = os.path.join(target_dir, rel_path) if rel_path != '.' else target_dir
                os.makedirs(target_subdir, exist_ok=True)
                
                # Create new filename with result_dir_name suffix
                filename_base, ext = os.path.splitext(file)
                new_filename = f"{filename_base}_{result_dir_name}{ext}"
                target_path = os.path.join(target_subdir, new_filename)
                
                # Copy the file with the new name
                shutil.copy2(source_path, target_path)