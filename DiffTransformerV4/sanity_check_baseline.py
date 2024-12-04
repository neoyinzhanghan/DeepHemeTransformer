from dataset import TensorStackDataModuleV4

feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
logit_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"
diff_data_path = (
    "/media/hdd3/neo/DiffTransformerV1DataMini/subsampled_split_diff_data.csv"
)


data_module = TensorStackDataModuleV4(
    feature_stacks_dir=feature_stacks_dir,
    logit_stacks_dir=logit_stacks_dir,
    diff_data_path=diff_data_path,
    batch_size=16,
    num_workers=8,
)

data_module.setup()

train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()
test_dataloader = data_module.test_dataloader()

# how many datapoints are in each split
num_train = len(train_dataloader.dataset)
num_val = len(val_dataloader.dataset)
num_test = len(test_dataloader.dataset)

print(f"Number of training datapoints: {num_train}")
print(f"Number of validation datapoints: {num_val}")
print(f"Number of test datapoints: {num_test}")


for idx, (feature_stack, logit_stack, diff_tensor) in enumerate(train_dataloader):
    print(f"Feature stack shape: {feature_stack.shape}")
    print(f"Logit stack shape: {logit_stack.shape}")
    print(f"Diff tensor shape: {diff_tensor.shape}")
