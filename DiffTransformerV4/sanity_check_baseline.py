from dataset import TensorStackDataModuleV4

feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
logit_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"
diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"


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

# how many datapoints are in each split
num_train = len(train_dataloader.dataset)
num_val = len(val_dataloader.dataset)

print(f"Number of training datapoints: {num_train}")
print(f"Number of validation datapoints: {num_val}")


for idx, (feature_stack, logit_stack, NPM, diff_tensor) in enumerate(train_dataloader):
    print(f"Feature stack shape: {feature_stack.shape}")
    print(f"Logit stack shape: {logit_stack.shape}")
    print(f"NPM shape: {NPM.shape}")
    print(f"Diff tensor shape: {diff_tensor.shape}")

    # take the sum of the logit stack. It has dimension [batch_size, num_cells, num_classes]
    sum_logits = logit_stack.sum(dim=1)

    # then divide by the sum of the NPM, the NPM has shape [batch_size, num_cells] and the sum output should have shape [batch_size, 1]
    sum_NPM = NPM.sum(dim=1).unsqueeze(1)

    # divide the sum_logits by the sum_NPM
    normalized_logits = sum_logits / sum_NPM

    print(f"Sum logits shape: {sum_logits.shape}")
    print(f"Sum NPM shape: {sum_NPM.shape}")
    print(f"Normalized logits shape: {normalized_logits.shape}")

    import sys
    sys.exit()
