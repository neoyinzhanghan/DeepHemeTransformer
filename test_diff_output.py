import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from DeepHemeTransformer import DeepHemeModule, load_model
from cell_dataloader import ImagePathDataset, custom_collate_fn, CellFeaturesDataModule
from BMAassumptions import index_map, BMA_final_classes


def one_hot_encode_and_average(ground_truth_probabilities):
    # Get the index of the maximum probability for each sample
    max_indices = ground_truth_probabilities.argmax(dim=1)

    # Create a one-hot encoded tensor based on these indices
    one_hot_encoded = F.one_hot(
        max_indices, num_classes=ground_truth_probabilities.shape[1]
    ).float()

    # Average across N to get the final probabilities
    average_one_hot = one_hot_encoded.mean(dim=0)

    return average_one_hot


def plot_probability_bar_chart(
    inputs, ground_truth_prob_tens, ground_truth_probabilities, save_path
):
    # Initialize an output tensor for the summed values
    N = inputs.shape[0]

    inputs = F.softmax(inputs, dim=1)
    outputs = torch.zeros(N, len(index_map), device=inputs.device)

    # Sum values according to the index map
    for new_idx, old_indices in index_map.items():
        for old_idx in old_indices:
            outputs[:, new_idx] += inputs[:, old_idx]

    average_probabilities = outputs.mean(dim=0)

    # Ensure lengths match the final class list length
    assert (
        len(average_probabilities)
        == len(ground_truth_probabilities)
        == len(BMA_final_classes)
    ), f"Length of predicted_probabilities and ground_truth_probabilities should be the same and equal to the length of BMA_final_classes. We got predicted_probabilities: {len(average_probabilities)}, ground_truth_probabilities: {len(ground_truth_probabilities)}, BMA_final_classes: {len(BMA_final_classes)}"

    # Averaging predicted probabilities across all samples
    avg_predicted_probabilities = average_probabilities.cpu().detach().numpy()

    # old_avg_predicted_probabilities = (
    #     ground_truth_prob_tens.mean(dim=0).cpu().numpy()
    # ) # this the non-one-hot encoded version

    old_avg_predicted_probabilities = (
        one_hot_encode_and_average(ground_truth_prob_tens).cpu().numpy()
    )

    ground_truth_probabilities = ground_truth_probabilities.cpu().numpy()

    # divide the ground truth probabilities by 100
    ground_truth_probabilities = ground_truth_probabilities / 100

    # Set seaborn style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("pastel")

    # Plotting
    x = range(len(BMA_final_classes))
    width = 0.25  # Adjusted width for three sets of bars

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot predicted probabilities
    ax.bar(
        x,
        avg_predicted_probabilities,
        width,
        label="Predicted Probabilities",
        color=palette[1],
    )

    # Plot ground truth probabilities with an offset
    ax.bar(
        [i + width for i in x],
        ground_truth_probabilities,
        width,
        label="Ground Truth Probabilities",
        color=palette[3],
    )

    # Plot old average predicted probabilities with a further offset
    ax.bar(
        [i + 2 * width for i in x],
        old_avg_predicted_probabilities,
        width,
        label="Old Avg Predicted Probabilities",
        color=palette[4],
    )

    # Labels, title, and legend
    ax.set_xlabel("Classes", fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)
    ax.set_title("Predicted vs Ground Truth Probabilities", fontsize=16, weight="bold")
    ax.set_xticks([i + width for i in x])
    ax.set_xticklabels(BMA_final_classes, rotation=45, ha="right", fontsize=12)
    ax.legend(fontsize=12)

    # Use tight layout and save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


model_checkpoint_path = "/home/greg/Documents/neo/DeepHemeTransformer/logs/tune_nov3/lr_3.698482555831662e-05_no_reg/version_0/checkpoints/epoch=4-step=375.ckpt"
metadata_path = "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
plot_save_dir = "/media/hdd3/neo/DeepHemeTransformerResults/diff_bar_plots"
os.makedirs(plot_save_dir, exist_ok=True)
model = load_model(model_checkpoint_path)
# move model to GPU
model = model.cuda()
model.eval()


cell_features_data_module = CellFeaturesDataModule(
    metadata_file=metadata_path, batch_size=32
)

cell_features_data_module.setup()

test_dataset = cell_features_data_module.test_dataloader()

for i, (features, logits, diff_tensors) in tqdm(
    enumerate(test_dataset), total=len(test_dataset)
):

    for j in range(len(features)):

        # move the feature to GPU
        feature = features[j]
        feature = feature.cuda()
        ground_truth_prob_tens = logits[j]

        inputs = model(feature)

        # move inputs to cpu
        inputs = inputs.cpu()

        # Get the ground truth probabilities
        ground_truth_probabilities = diff_tensors[j]

        # move the ground truth probabilities to cpu
        ground_truth_probabilities = ground_truth_probabilities.cpu()

        save_path = os.path.join(plot_save_dir, f"bar_chart_{i}_{j}.png")
        plot_probability_bar_chart(
            inputs=inputs,
            ground_truth_prob_tens=ground_truth_prob_tens,
            ground_truth_probabilities=ground_truth_probabilities,
            save_path=save_path,
        )
