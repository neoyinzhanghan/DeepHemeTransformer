import os
import torch
import pandas as pd
from BMAassumptions import BMA_final_classes

# from CELoss import custom_cross_entropy_loss
from AR_acc import AR_acc, A_acc, R_acc, Class_AR_acc, Class_A_acc, Class_R_acc
from L2Loss import MyL2Loss
from TRL2Loss import MyTRL2Loss

ar_acc_fn = AR_acc()
a_acc_fn = A_acc()
r_acc_fn = R_acc()
l2_loss_fn = MyL2Loss()
tr_l2_loss_fn = MyTRL2Loss()

diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"
pipeline_diff_path = "/media/hdd3/neo/DiffTransformerV1DataMini/pipeline_diff.csv"

diff_data = pd.read_csv(diff_data_path)
pipeline_diff = pd.read_csv(pipeline_diff_path)

# print the number of rows in the differential data
print(f"Number of rows in differential data: {len(diff_data)}")

# print the number of rows in the pipeline_diff data
print(f"Number of rows in pipeline_diff data: {len(pipeline_diff)}")

# iterate over the rows of the pipeline_diff dataframe

loss_list = []

ar_acc_sum = 0
a_acc_sum = 0
r_acc_sum = 0
l2_loss_sum = 0
trl2_loss_sum = 0

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
        print(f"result_dir_name: {result_dir_name} not found in diff_data")
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

    ar_acc = ar_acc_fn(diff_tens, pipeline_diff_tens)
    a_acc = a_acc_fn(diff_tens, pipeline_diff_tens)
    r_acc = r_acc_fn(diff_tens, pipeline_diff_tens)

    ar_acc_sum += ar_acc
    a_acc_sum += a_acc
    r_acc_sum += r_acc

    l2_loss_sum += l2_loss_fn(diff_tens, pipeline_diff_tens)
    trl2_loss_sum += tr_l2_loss_fn(diff_tens, pipeline_diff_tens)

    for final_class in BMA_final_classes:
        class_ar_acc_fn = Class_AR_acc(final_class)
        class_a_acc_fn = Class_A_acc(final_class)
        class_r_acc_fn = Class_R_acc(final_class)
        class_ar_acc = class_ar_acc_fn(diff_tens, pipeline_diff_tens)
        class_a_acc = class_a_acc_fn(diff_tens, pipeline_diff_tens)
        class_r_acc = class_r_acc_fn(diff_tens, pipeline_diff_tens)

        class_ar_acc_dct[final_class] += class_ar_acc
        class_a_acc_dct[final_class] += class_a_acc
        class_r_acc_dct[final_class] += class_r_acc

    tot_num += 1

    print(f"tot_num: {tot_num}")

ar_acc_avg = ar_acc_sum / tot_num
a_acc_avg = a_acc_sum / tot_num
r_acc_avg = r_acc_sum / tot_num
l2_loss_avg = l2_loss_sum / tot_num
trl2_loss_avg = trl2_loss_sum / tot_num

for final_class in BMA_final_classes:
    class_ar_acc_dct[final_class] = class_ar_acc_dct[final_class] / tot_num
    class_a_acc_dct[final_class] = class_a_acc_dct[final_class] / tot_num
    class_r_acc_dct[final_class] = class_r_acc_dct[final_class] / tot_num

print(f"AR acc: {ar_acc_avg}")
print(f"A acc: {a_acc_avg}")
print(f"R acc: {r_acc_avg}")
print(f"L2 loss: {l2_loss_avg}")
print(f"TRL2 loss: {trl2_loss_avg}")

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
