import os
import pandas as pd
from tqdm import tqdm
from train_plasma_cell import load_model, predict_image

original_cells_dir = "/media/hdd2/neo/high_plasma_cell_slides_cells"
new_cells_dir = "/media/hdd2/neo/high_plasma_cell_slides_cells_relabeled"

os.makedirs(new_cells_dir, exist_ok=True)

differential_group_dict = {
    "blasts and blast-equivalents": ["M1"],
    "promyelocytes": ["M2"],
    "myelocytes": ["M3"],
    "metamyelocytes": ["M4"],
    "neutrophils/bands": ["M5", "M6"],
    "monocytes": ["MO2"],
    "eosinophils": ["E1", "E4"],
    "erythroid precursors": ["ER1", "ER2", "ER3", "ER4"],
    "lymphocytes": ["L2"],
    "plasma cells": ["L4"],
}

# grouped_label_to_index
grouped_label_to_index = {
    "blasts and blast-equivalents": 0,
    "promyelocytes": 1,
    "myelocytes": 2,
    "metamyelocytes": 3,
    "neutrophils/bands": 4,
    "monocytes": 5,
    "eosinophils": 6,
    "erythroid precursors": 7,
    "lymphocytes": 8,
    "plasma cells": 9,
    "skippocytes": 10,
}

# grouped_labels
grouped_labels = list(grouped_label_to_index.keys())

cell_metadata_dict = {
    "cell_result_dir": [],
    "cell_path": [],
    "original_label": [],
    "new_label": [],
}


model = load_model(
    "/home/greg/Documents/neo/DeepHemeRetrain/lightning_logs/1/version_0/checkpoints"
)

# get the list of subdirectories in the original_cells_dir
result_dirs = [
    os.path.join(original_cells_dir, d) for d in os.listdir(original_cells_dir)
]

for result_dir in tqdm(result_dirs, desc="Processing Slides"):
    # recursively find the paths of all the jpg files in the cells_dir
    cells_files = []
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith(".jpg") and "YOLO" not in file:
                cells_files.append(os.path.join(root, file))

    for cell_file in tqdm(cells_files, desc="Processing Cells"):
        predicted_class, probabilities = predict_image(model, cell_file)
        new_label = grouped_label_to_index[predicted_class]
        cell_metadata_dict["cell_result_dir"].append(result_dir)
        cell_metadata_dict["cell_path"].append(cell_file)
        cell_metadata_dict["original_label"].append(predicted_class)
        cell_metadata_dict["new_label"].append(new_label)

# save the cell_metadata_dict to a csv file
cell_metadata_df = pd.DataFrame(cell_metadata_dict)
cell_metadata_df.to_csv(
    os.path.join(new_cells_dir, "high_plasma_cell_relabel_metadata.csv"), index=False
)
