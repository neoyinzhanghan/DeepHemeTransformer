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
        self.criterion = nn.KLDivLoss(
            reduction="batchmean"
        )  # KLDivLoss expects log-probabilities and probabilities

    def forward(self, inputs_list, targets_list):
        """
        Args:
            inputs_list (list of torch.Tensor): List of tensors containing probability distributions from the model.
            targets_list (list of torch.Tensor): List of tensors containing ground truth probability distributions.

        Returns:
            torch.Tensor: Mean loss across the batch.
        """
        # Initialize a list to store the individual losses
        batch_losses = []

        for inputs, targets in zip(inputs_list, targets_list):
            # Initialize an output tensor for the summed values
            N = inputs.shape[0]

            inputs = F.softmax(inputs, dim=1)
            outputs = torch.zeros(N, len(self.index_map), device=inputs.device)

            # Sum values according to the index map
            for new_idx, old_indices in self.index_map.items():
                for old_idx in old_indices:
                    outputs[:, new_idx] += inputs[:, old_idx]

            average_probabilities = outputs.mean(dim=0)

            # Apply softmax for additional smoothing on model outputs (single vector)
            smoothed_probabilities = F.softmax(average_probabilities, dim=0)

            # assert that targets is of shape [len(index_map)]
            assert targets.shape[0] == len(
                self.index_map
            ), f"Expected targets to have shape [{len(index_map)}], but got [{targets.shape}]."

            # divide the ground truth probabilities by 100
            targets = targets / 100

            # Apply softmax to the target tensor to get smoothed probabilities (single vector)
            smoothed_targets = nn.Softmax(dim=0)(targets.squeeze())

            # Apply log to the model probabilities since KLDivLoss expects log-probabilities
            log_probabilities = torch.log(
                smoothed_probabilities + 1e-8
            )  # Adding a small epsilon for numerical stability

            # Compute the KL divergence loss for each sample
            loss = self.criterion(
                log_probabilities, smoothed_targets
            )  # Compare with smoothed targets

            # Append the loss for each sample to the batch_losses list
            batch_losses.append(loss)

        # Average the losses across the batch dimension
        return torch.mean(torch.stack(batch_losses))  # Final average across the batch


class RegularizedDifferentialLoss(nn.Module):
    def __init__(self, reg_lambda=0.1, index_map=index_map):
        super(RegularizedDifferentialLoss, self).__init__()
        self.reg_lambda = reg_lambda
        self.average_ce_loss = AvgCELoss()
        self.differential_loss = GroupedLossWithIndexMap(index_map)

    def forward(self, outputs_list, logits_list, differentials_list):

        # print(type(outputs_list))

        # print(len(outputs_list))
        # print(outputs_list[0].shape[0])

        # print(type(logits_list))

        # print(len(logits_list))
        # print(logits_list[0].shape)

        # print(type(differentials_list))

        # print(len(differentials_list))
        # print

        # import sys

        # sys.exit() # TODO to remove only for debugging

        cell_classes = [logits.argmax(dim=1) for logits in logits_list]

        # Compute the average cross-entropy loss
        ce_loss = self.average_ce_loss(outputs_list, cell_classes)

        # Compute the differential loss
        diff_loss = self.differential_loss(outputs_list, differentials_list)

        # Combine the losses
        # total_loss = ce_loss
        total_loss = diff_loss + self.reg_lambda * ce_loss

        return total_loss
