import os
import pandas as pd

data_metadata_path = ""
save_dir = ""
test_proportion = 0.1

# first open the metadata file
metadata = pd.read_csv(data_metadata_path)
