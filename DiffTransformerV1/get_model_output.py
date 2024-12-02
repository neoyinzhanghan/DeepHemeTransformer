from DiffTransformerV1.DiffTransformerV1 import MultiHeadAttentionClassifierPL
from dataset import TensorStackDataModule

model_ckpt_path = "/home/greg/Documents/neo/DeepHemeTransformer/DiffTransformerV1/lightning_logs/multihead_attention/version_5/checkpoints/epoch=499-step=500.ckpt"
feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
diff_data_path = (
    "/media/hdd3/neo/DiffTransformerV1DataMini/subsampled_split_diff_data.csv"
)

data_module = TensorStackDataModule(
    feature_stacks_dir=feature_stacks_dir,
    diff_data_path=diff_data_path,
    batch_size=5,
    num_workers=5,
)

model = MultiHeadAttentionClassifierPL.load_from_checkpoint(model_ckpt_path, map_location="cuda")

train_loader = data_module.train_dataloader()

model.eval()

# Get the model output
for batch in train_loader:
    x, y = batch

    # make sure that x and y are on the same device as the model
    x = x.to(model.device)
    y = y.to(model.device)

    logits = model(x)

    print(logits.shape)
    print(y.shape)
    break
