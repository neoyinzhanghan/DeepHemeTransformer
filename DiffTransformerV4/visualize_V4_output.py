from DiffTransformerV4 import MultiHeadAttentionClassifierPL
from dataset import TensorStackDataModuleV4


model_checkpoint_path = "/home/greg/Documents/neo/DeepHemeTransformer/DiffTransformerV4/lightning_logs_V4/multihead_attention_classifier/version_0/checkpoints/epoch=49-step=50.ckpt"
model = MultiHeadAttentionClassifierPL.load_from_checkpoint(model_checkpoint_path)
feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
logit_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"
diff_data_path = (
    "/media/hdd3/neo/DiffTransformerV1DataMini/subsampled_split_diff_data.csv"
)

data_module = TensorStackDataModuleV4(
    feature_stacks_dir=feature_stacks_dir,
    logit_stacks_dir=logit_stacks_dir,
    diff_data_path=diff_data_path,
    batch_size=1,
    num_workers=8,
)
data_module.setup()

train_loader = data_module.train_dataloader()

for batch in train_loader:
    feature_stack, logit_stack, NPM, y = batch

    model.eval()
    # move the model to cuda
    model = model.to("cuda")
    # move everything to the cuda
    feature_stack = feature_stack.to(model.device)
    logit_stack = logit_stack.to(model.device)
    NPM = NPM.to(model.device)

    y_hat = model(feature_stack, logit_stack, NPM)
    y_hat_baseline = model.baseline_forward(logit_stack)

    print(f"Shape of y_hat: {y_hat.shape}")
    print(f"Shape of y_hat_baseline: {y_hat_baseline.shape}")
    print(f"Shape of y: {y.shape}")

    import sys

    sys.exit()
