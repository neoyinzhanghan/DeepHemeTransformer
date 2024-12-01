import os
import torch
import pandas as pd
from BMAassumptions import BMA_final_classes
from CELoss import custom_cross_entropy_loss

diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"
pipeline_diff_path = "/media/hdd3/neo/DiffTransformerV1DataMini/pipeline_diff.csv"

diff_data = pd.read_csv(diff_data_path)
pipeline_diff = pd.read_csv(pipeline_diff_path)

# iterate over the rows of the pipeline_diff dataframe

loss_list = []

for index, row in pipeline_diff.iterrows():
    # get the result_dir_name
    result_dir_name = row["result_dir_name"]
    # get the row in diff_data that has the same result_dir_name
    diff_row = diff_data[diff_data["result_dir_name"] == result_dir_name]

    diff_tens_list, pipeline_diff_tens_list = [], []
    for final_class in BMA_final_classes:
        diff_tens_list.append(diff_row[final_class])
        pipeline_diff_tens_list.append(row[final_class])

    diff_tens = torch.tensor(diff_tens_list)
    pipeline_diff_tens = torch.tensor(pipeline_diff_tens_list)

    # squeeze the tensors
    diff_tens = diff_tens.unsqueeze(0)
    pipeline_diff_tens = pipeline_diff_tens.unsqueeze(0)

    loss_item = custom_cross_entropy_loss(diff_tens, pipeline_diff_tens)

    loss_list.append(loss_item)

print(f"Average baseline loss: {sum(loss_list) / len(loss_list)}")
