import pandas as pd
from get_copath_data import get_diff, get_path_data

metadata_path = "/media/hdd3/neo/DiffTransformerV1DataMini/wsi_metadata.csv"
metadata = pd.read_csv(metadata_path)

# get the accession_numbers as a list of strings
accession_numbers = metadata["accession_number"].tolist()

# print the first 5 accession numbers
print(accession_numbers[:5])

# print the length of the accession_numbers list
print(f"Number of accession numbers: {len(accession_numbers)}")

path_df = get_path_data(accession_numbers)

# print the number of rows in the path data
print(f"Number of rows in path data: {len(path_df)}")

diff = get_diff(path_df)

# print the number of rows in the differential data
print(f"Number of rows in differential data: {len(diff)}")

# remove the columns named text_data_final, part_description
diff = diff.drop(["text_data_final", "part_description"], axis=1)

# rename specnum_formatted to accession_number
diff = diff.rename(columns={"specnum_formatted": "accession_number"})

# traverse through the rows of diff
for i, row in diff.iterrows():
    # get the accession number
    accession_number = row["accession_number"]
    # get the accession number from the metadata
    metadata_row = metadata[metadata["accession_number"] == accession_number]
    # get the result_dir_name
    result_dir_name = metadata_row["result_dir_name"].values[0]
    # assign the result_dir_name to the result_dir_name column in the diff dataframe
    diff.loc[i, "result_dir_name"] = result_dir_name

    # for each column in the row, if the entry is a number, divide by 100
    for col in diff.columns:
        if diff[col].dtype == "float64":
            diff.loc[i, col] = diff.loc[i, col] / 100

# save the differential data to a csv file
diff.to_csv("/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv", index=False)
