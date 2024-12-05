import pandas as pd
from get_copath_data import get_diff, get_path_data
from BMAassumptions import BMA_final_classes

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

# replace all nan values with 0 in diff
diff = diff.fillna(0)

# sum together the columns named blasts and blast-equivalents into a new column named "blasts and blast-equivalents"
diff["blasts and blast-equivalents"] = (
    diff["blasts"] + diff["blast-equivalents"] + diff["promyelocytes"]
)

# print the top 5 rows of the diff dataframe at the blasts column
# print("Blasts")
# print(diff["blasts"].head())
# print("Blast Equivalents")
# print(diff["blast-equivalents"].head())
# print("Promyelocytes")
# print(diff["promyelocytes"].head())
# print("Blasts and Blast Equivalents")
# print(diff["blasts and blast-equivalents"].head()) # TODO REMOVE. this is for debugging only


# # remove the columns named blasts and blast-equivalents
# diff = diff.drop(["blasts", "blast-equivalents", "promyelocytes"], axis=1)


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

    total = 0

    # print the name of all the columns in the row
    print(diff.columns)

    # for each column in the row, if the entry is a number, divide by 100
    for col in diff.columns:
        if diff[col].dtype == "float64":
            diff.loc[i, col] = diff.loc[i, col] / 100
            # if nan, replace with 0
            if pd.isna(diff.loc[i, col]):
                diff.loc[i, col] = 0
            total += diff.loc[i, col]

    # assign the total to the total column in the diff dataframe
    diff.loc[i, "total"] = total
    print(f"Accession number: {accession_number}, Total: {total}")

# remove all rows where total is <0.9 also report how many rows were removed
good_diff = diff[diff["total"] >= 0.9]
removed_diff = diff[diff["total"] < 0.9]

# # renormalize the values of the BMA_final_classes columns to sum to 1 by dividing by the total
# for col in BMA_final_classes:
#     good_diff.iloc[:, good_diff.columns.get_loc(col)] = (
#         good_diff[col] / good_diff["total"]
#     ) # TODO DEPRECATED WE WILL NO LONGER RENORMALIZE THE BMA_final_classes columns

# remove the total column
good_diff = good_diff.drop("total", axis=1)

# # verify that the sum of the BMA_final_classes columns is 1 within a tolerance of 1e-6
# assert all(
#     abs(good_diff[BMA_final_classes].sum(axis=1) - 1) < 1e-6
# ), "The sum of the BMA_final_classes columns is not 1"

print(f"Number of rows removed: {len(removed_diff)}")
print(f"Number of rows remaining: {len(good_diff)}")

# save the differential data to a csv file
good_diff.to_csv("/media/hdd3/neo/DiffTransformerV1DataMini/diff_data.csv", index=False)
