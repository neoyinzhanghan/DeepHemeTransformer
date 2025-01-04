import os
import pandas as pd

result_dir_path = "/media/hdd4/neo/results_dir"
pipeline_run_history_path = "/media/hdd4/neo/results_dir/pipeline_run_history.csv"
diff_data_path = "/media/hdd3/neo/diff_data_processed.csv"

# open the diff_data_processed.csv file
diff_data = pd.read_csv(diff_data_path)

# replace na values with 0
diff_data.fillna(0, inplace=True)

threshold = 50
# find the rows where the column plasma cells is >= 75
high_plasma_cell_slides = diff_data[diff_data["plasma cells"] >= threshold]

# print the number of rows where the column plasma cells is >= 75
print(f"Total number of rows: {len(diff_data)}")
print(f"Number of rows where plasma cells >= {threshold}: {len(high_plasma_cell_slides)}")

# save the high_plasma_cell_slides to a csv file at /media/hdd3/neo/high_plasma_cell_slides.csv
high_plasma_cell_slides.to_csv("/media/hdd3/neo/high_plasma_cell_slides.csv", index=False)
