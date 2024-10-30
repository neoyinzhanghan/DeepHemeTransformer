import torch
import torch.nn as nn
import torch.nn.functional as F
from BMAassumptions import index_map


class AvgCELoss(nn.Module):
    def __init__(self):
        super(AvgCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(
            reduction="none"
        )  # Set reduction to 'none' to calculate per-sample loss

    def forward(self, inputs_list, targets_list):
        # Initialize a list to store the individual losses
        batch_losses = []

        # print(f"Length of inputs_list: {len(inputs_list)}")
        # print(f"Length of targets_list: {len(targets_list)}")
        # print(targets_list) # TODO to remove only for debugging

        for inputs, targets in zip(inputs_list, targets_list):

            # print(inputs.shape, targets.shape)
            # print(f"Shape of inputs: {inputs.shape}")
            # print(f"Shape of targets: {targets.shape}") # TODO to remove only for debugging

            # import sys

            # sys.exit()

            # Compute the cross-entropy loss for each sample in the batch
            losses = self.criterion(inputs, targets)

            # Average the loss across N for each sample
            avg_loss = losses.mean()  # Shape: []

            # Append the average loss for each sample to the batch_losses list
            batch_losses.append(avg_loss)

        # Average the losses across the batch dimension
        return torch.mean(torch.stack(batch_losses))  # Final average across the batch


class GroupedLossWithIndexMap(nn.Module):
    def __init__(self, index_map):
        super(GroupedLossWithIndexMap, self).__init__()
        self.index_map = index_map
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, inputs_list, targets_list):
        # Initialize a list to store the individual losses
        batch_losses = []

        for inputs, targets in zip(inputs_list, targets_list):
            # Initialize an output tensor for the summed values
            N, _ = inputs.shape
            outputs = torch.zeros(N, len(self.index_map), device=inputs.device)

            # Sum values according to the index map
            for new_idx, old_indices in self.index_map.items():
                for old_idx in old_indices:
                    outputs[:, new_idx] += inputs[:, old_idx]

            # Normalize to get a probability distribution
            sum_outputs = outputs.sum(
                dim=-1, keepdim=True
            )  # Compute the sum across the last dimension
            probabilities = outputs / sum_outputs  # Basic normalization

            # Average across N to get shape [11]
            avg_probabilities = probabilities.mean(dim=0)  # Average over N

            # Compute the cross-entropy loss for each sample
            loss = self.criterion(avg_probabilities.log(), targets)

            # Append the loss for each sample to the batch_losses list
            batch_losses.append(loss)

        # Average the losses across the batch dimension
        return torch.mean(torch.stack(batch_losses))  # Final average across the batch


class RegularizedDifferentialLoss(nn.Module):
    def __init__(self, reg_lambda=0.1, index_map=None):
        super(RegularizedDifferentialLoss, self).__init__()
        self.reg_lambda = reg_lambda
        self.index_map = index_map
        self.differential_loss = GroupedLossWithIndexMap(index_map)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits_list, differentials_list):
        """
        Args:
            logits_list (list of torch.Tensor): List of tensors containing raw model outputs (logits) for each batch.
            differentials_list (list of torch.Tensor): List of tensors containing ground truth differentials.

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Softmax logits to get probabilities
        outputs_list = [F.softmax(logits, dim=1) for logits in logits_list]

        # Softmax differentials to get probability distributions
        differentials_list = [
            F.softmax(differential, dim=1) for differential in differentials_list
        ]

        # Compute the cell classes (argmax of logits to determine predicted class)
        cell_classes = [logits.argmax(dim=1) for logits in logits_list]

        # Compute the average cross-entropy loss over all batches in logits_list
        ce_loss = sum(
            self.cross_entropy_loss(logits, cell_class)
            for logits, cell_class in zip(logits_list, cell_classes)
        ) / len(logits_list)

        # Compute the differential loss using the provided differentials_list
        diff_loss = self.differential_loss(outputs_list, differentials_list)

        # Combine the losses using the regularization weight
        total_loss = diff_loss + self.reg_lambda * ce_loss

        return total_loss
