import os
import matplotlib.pyplot as plt
from DiffTransformerV1 import MultiHeadAttentionClassifierPL
from dataset import TensorStackDataModule

model_ckpt_path = "/home/greg/Documents/neo/DeepHemeTransformer/DiffTransformerV1/lightning_logs/multihead_attention/version_0/checkpoints/epoch=499-step=500.ckpt"
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

model = MultiHeadAttentionClassifierPL.load_from_checkpoint(
    model_ckpt_path, map_location="cuda"
)
data_module.setup()
train_loader = data_module.train_dataloader()

model.eval()

# Get the model output
for batch in train_loader:
    x, y = batch

    # make sure that x and y are on the same device as the model
    x = x.to(model.device)
    y = y.to(model.device)

    logits = model(x)

    break

save_dir = "tmp_plots"
os.makedirs(save_dir, exist_ok=True)

for i in range(len(logits)):
    logit_list = logits[i].tolist()
    y_list = y[i].tolist()

    print(f"Example {i}")
    print(f"Logits: {logit_list}")
    print(f"Labels: {y_list}")
    # Creating the bar plot
    x_labels = [
        f"Item {i+1}" for i in range(len(logit_list))
    ]  # Create labels for the x-axis
    x = range(len(logit_list))  # x-axis positions

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the bars
    width = 0.4  # Width of the bars
    ax.bar([pos - width / 2 for pos in x], logit_list, width, label="Logits")
    ax.bar([pos + width / 2 for pos in x], y_list, width, label="Y Values")

    # Adding labels and title
    ax.set_xlabel("Items", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.set_title("Comparison of Logit and Y Values", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{save_dir}/example_{i}.png")
