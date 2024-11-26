import pandas as pd
from BMAassumptions import BMA_final_classes

diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv"
pipeline_diff_path = "/media/hdd3/neo/DiffTransformerV1DataMini/pipeline_diff.csv"

diff_data = pd.read_csv(diff_data_path)
pipeline_diff = pd.read_csv(pipeline_diff_path)

paired_diff_data = {
    "result_dir_name": [],
}

for BMA_final_class in BMA_final_classes:
    paired_diff_data[BMA_final_class + " (GT)"] = []
    paired_diff_data[BMA_final_class + " (Pipeline)"] = []

for i, row in diff_data.iterrows():
    result_dir_name = row["result_dir_name"]
    diff = row[BMA_final_classes].values

    pipeline_diff_row = pipeline_diff[
        pipeline_diff["result_dir_name"] == result_dir_name
    ]

    paired_diff_data["result_dir_name"].append(result_dir_name)

    for BMA_final_class in BMA_final_classes:
        paired_diff_data[BMA_final_class + " (Pipeline)"].append(
            pipeline_diff_row[BMA_final_class].values[0]
        )
        paired_diff_data[BMA_final_class + " (GT)"].append(
            diff[BMA_final_classes.index(BMA_final_class)]
        )

paired_diff_df = pd.DataFrame(paired_diff_data)
paired_diff_df.to_csv(
    "/media/hdd3/neo/DiffTransformerV1DataMini/paired_diff.csv", index=False
)
