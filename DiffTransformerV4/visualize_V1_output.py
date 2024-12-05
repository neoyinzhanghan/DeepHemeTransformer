import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from DiffTransformerV1 import MultiHeadAttentionClassifierPL
from dataset import TensorStackDataModuleV4
from BMAassumptions import BMA_final_classes


model_checkpoint_path = "/home/greg/Documents/neo/DeepHemeTransformer/DiffTransformerV4/lightning_logs_test/multihead_attention_classifier/version_1/checkpoints/epoch=99-step=900.ckpt"
model = MultiHeadAttentionClassifierPL.load_from_checkpoint(model_checkpoint_path)
feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
logit_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"
diff_data_path = (
    "/media/hdd3/neo/DiffTransformerV1DataMini/subsampled_split_diff_data.csv"
)

plot_save_dir = "tmp_plots_V1"
os.makedirs(plot_save_dir, exist_ok=True)


def make_barplot(y_hat, y_hat_baseline, y, save_path):
    """
    Creates a bar plot comparing y_hat, y_hat_baseline, and y values for 9 classes.

    Args:
        y_hat (np.ndarray): Array of predicted values for a model.
        y_hat_baseline (np.ndarray): Array of predicted values for a baseline model.
        y (np.ndarray): Array of ground truth values.
        save_path (str): Path to save the plot.
    """
    x = np.arange(len(BMA_final_classes))  # x positions

    width = 0.25  # Width of each bar
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    ax.bar(x - width, y_hat, width, label="y_hat", color="blue")
    ax.bar(x, y_hat_baseline, width, label="y_hat_baseline", color="orange")
    ax.bar(x + width, y, width, label="y", color="green")

    # Add labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(BMA_final_classes, rotation=45, ha="right")
    ax.set_ylabel("Values")
    ax.set_title("Comparison of y_hat, y_hat_baseline, and y")
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


data_module = TensorStackDataModuleV4(
    feature_stacks_dir=feature_stacks_dir,
    logit_stacks_dir=logit_stacks_dir,
    diff_data_path=diff_data_path,
    batch_size=1,
    num_workers=8,
)
data_module.setup()

train_loader = data_module.train_dataloader()

for batch_idx, batch in tqdm(enumerate(train_loader), desc="Visualizing Results"):
    feature_stack, logit_stack, NPM, y = batch

    model.eval()
    # move the model to cuda
    model = model.to("cuda")
    # move everything to the cuda
    feature_stack = feature_stack.to(model.device)
    logit_stack = logit_stack.to(model.device)
    NPM = NPM.to(model.device)

    y_hat = model(feature_stack)
    y_hat_baseline = model.baseline_forward(logit_stack)

    # the shape is [1, 9], reshape to [9] and turn to a np array
    y_hat = y_hat[0].detach().cpu().numpy()
    y_hat_baseline = y_hat_baseline[0].detach().cpu().numpy()
    y = y[0].detach().cpu().numpy()

    print(f"Shape of y_hat: {y_hat.shape}")
    print(f"Shape of y_hat_baseline: {y_hat_baseline.shape}")
    print(f"Shape of y: {y.shape}")

    save_path = f"{plot_save_dir}/example_{batch_idx}.png"

    make_barplot(y_hat, y_hat_baseline, y, save_path)

    import sys

    sys.exit()
