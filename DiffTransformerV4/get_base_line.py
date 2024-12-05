import os
import torch
import pandas as pd
from BMAassumptions import BMA_final_classes

# from CELoss import custom_cross_entropy_loss
from AR_acc import AR_acc, A_acc, R_acc, Class_AR_acc, Class_A_acc, Class_R_acc

ar_acc = AR_acc()
a_acc = A_acc()
r_acc = R_acc()

diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"
pipeline_diff_path = "/media/hdd3/neo/DiffTransformerV1DataMini/pipeline_diff.csv"

diff_data = pd.read_csv(diff_data_path)
pipeline_diff = pd.read_csv(pipeline_diff_path)

# iterate over the rows of the pipeline_diff dataframe

loss_list = []

ar_acc_sum = 0
a_acc_sum = 0
r_acc_sum = 0

class_ar_acc_dct = {}
class_a_acc_dct = {}
class_r_acc_dct = {}

for final_class in BMA_final_classes:
    class_ar_acc_dct[final_class] = 0
    class_a_acc_dct[final_class] = 0
    class_r_acc_dct[final_class] = 0

tot_num = 0

for index, row in pipeline_diff.iterrows():
    # get the result_dir_name
    result_dir_name = row["result_dir_name"]
    # get the row in diff_data that has the same result_dir_name
    diff_row = diff_data[diff_data["result_dir_name"] == result_dir_name]

    if len(diff_row) == 0:
        continue
    # assert len(diff_row) == 1, f"There should be exactly one row in diff_data with the same result_dir_name. num rows: {len(diff_row)}"

    diff_tens_list, pipeline_diff_tens_list = [], []
    for final_class in BMA_final_classes:

        # print("final_class")
        # print(final_class)

        # print("diff_row[final_class]")
        # print(diff_row[final_class].iloc[0])

        # print("row[final_class]")
        # print(row[final_class])

        # import sys

        # sys.exit()

        diff_tens_list.append(diff_row[final_class].iloc[0])
        pipeline_diff_tens_list.append(row[final_class])

    diff_tens = torch.tensor(diff_tens_list)
    pipeline_diff_tens = torch.tensor(pipeline_diff_tens_list)

    # squeeze the tensors
    diff_tens = diff_tens.unsqueeze(0)
    pipeline_diff_tens = pipeline_diff_tens.unsqueeze(0)

    ar_acc = AR_acc(diff_tens, pipeline_diff_tens)
    a_acc = A_acc(diff_tens, pipeline_diff_tens)
    r_acc = R_acc(diff_tens, pipeline_diff_tens)

    ar_acc_sum += ar_acc
    a_acc_sum += a_acc
    r_acc_sum += r_acc

    for final_class in BMA_final_classes:
        class_ar_acc = Class_AR_acc(final_class, diff_tens, pipeline_diff_tens)
        class_a_acc = Class_A_acc(final_class, diff_tens, pipeline_diff_tens)
        class_r_acc = Class_R_acc(final_class, diff_tens, pipeline_diff_tens)

        class_ar_acc_dct[final_class] += class_ar_acc
        class_a_acc_dct[final_class] += class_a_acc
        class_r_acc_dct[final_class] += class_r_acc

    tot_num += 1

ar_acc_avg = ar_acc_sum / tot_num
a_acc_avg = a_acc_sum / tot_num
r_acc_avg = r_acc_sum / tot_num

for final_class in BMA_final_classes:
    class_ar_acc_dct[final_class] = class_ar_acc_dct[final_class] / tot_num
    class_a_acc_dct[final_class] = class_a_acc_dct[final_class] / tot_num
    class_r_acc_dct[final_class] = class_r_acc_dct[final_class] / tot_num

print(f"AR acc: {ar_acc_avg}")
print(f"A acc: {a_acc_avg}")
print(f"R acc: {r_acc_avg}")

for final_class in BMA_final_classes:
    print(
        f"Class {final_class} AR acc: {class_ar_acc_dct[
        final_class]}"
    )
    print(
        f"Class {final_class} A acc: {class_a_acc_dct[
        final_class]}"
    )
    print(
        f"Class {final_class} R acc: {class_r_acc_dct[
        final_class]}"
    )
