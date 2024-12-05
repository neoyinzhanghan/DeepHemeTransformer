from dataset import TensorStackDataModuleV4
from L2Loss import MyL2Loss
from TRL2Loss import MyTRL2Loss
from AR_acc import AR_acc
from DiffTransformerV4 import MultiHeadAttentionClassifierPL

feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
logit_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"
diff_data_path = (
    "/media/hdd3/neo/DiffTransformerV1DataMini/subsampled_split_diff_data.csv"
)

# initialize a difftransformer model
model = MultiHeadAttentionClassifierPL(
    d_model=2048, num_heads=1, num_classes=9, use_flash_attention=True
)


data_module = TensorStackDataModuleV4(
    feature_stacks_dir=feature_stacks_dir,
    logit_stacks_dir=logit_stacks_dir,
    diff_data_path=diff_data_path,
    batch_size=5,  # 16,
    num_workers=8,
)

data_module.setup()

train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()

l2_loss_fn = MyL2Loss()
trl2_loss_fn = MyTRL2Loss()
ar_acc_fn = AR_acc()

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

    l2_loss = l2_loss_fn(diff_tensor, normalized_logits)
    trl2_loss = trl2_loss_fn(diff_tensor, normalized_logits)
    ar_acc = ar_acc_fn(diff_tensor, normalized_logits)

    print(f"L2 Loss: {l2_loss}")
    print(f"TR L2 Loss: {trl2_loss}")
    print(f"AR Accuracy: {ar_acc}")

    predicted_diff = model(feature_stack, logit_stack, NPM)

    print(f"Predicted diff shape: {predicted_diff.shape}")

    predicted_l2_loss = l2_loss_fn(diff_tensor, predicted_diff)
    predicted_trl2_loss = trl2_loss_fn(diff_tensor, predicted_diff)
    predicted_ar_acc = ar_acc_fn(diff_tensor, predicted_diff)

    print(f"Predicted L2 Loss: {predicted_l2_loss}")
    print(f"Predicted TR L2 Loss: {predicted_trl2_loss}")
    print(f"Predicted AR Accuracy: {predicted_ar_acc}")

    print(predicted_diff[0, :])
    print(normalized_logits[0, :])
    print(diff_tensor[0, :])

    import sys

    sys.exit()
