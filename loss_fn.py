import torch
import torch.nn as nn
from BMAassumptions import index_map


class AvgCELoss(nn.Module):
    def __init__(self):
        super(AvgCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(
            reduction="none"
        )  # Set reduction to 'none' to calculate per-sample loss

    def forward(self, inputs, targets):
        # Check the shape of inputs and targets
        if inputs.dim() != 3 or targets.dim() != 2 or targets.size(1) != inputs.size(1):
            raise ValueError(
                "Input should be of shape [b, N, 23] and targets should be of shape [b, N]"
            )

        # Reshape inputs to [b*N, 23] for cross-entropy loss calculation
        inputs_reshaped = inputs.view(-1, inputs.size(-1))  # Shape: [b*N, 23]

        # Flatten targets to match the reshaped inputs
        targets_flattened = targets.view(-1)  # Shape: [b*N]

        # Compute the cross-entropy loss for each sample in the batch
        losses = self.criterion(inputs_reshaped, targets_flattened)

        # Reshape losses back to [b, N]
        losses = losses.view(inputs.size(0), -1)  # Shape: [b, N]

        # Average the loss across N for each batch
        avg_loss = losses.mean(dim=1)  # Shape: [b]

        # Average the losses across the batch dimension
        return avg_loss.mean()  # Final average across batches


class GroupedLossWithIndexMap(nn.Module):
    def __init__(self, index_map):
        super(GroupedLossWithIndexMap, self).__init__()
        self.index_map = index_map
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, inputs, targets):
        # Check input shapes
        if (
            inputs.dim() != 3
            or targets.dim() != 2
            or targets.size(1) != len(self.index_map)
        ):
            raise ValueError(
                "Input should be of shape [b, N, 23] and targets should be of shape [b, 11]"
            )

        # Initialize an output tensor for the summed values
        b, N, _ = inputs.shape
        outputs = torch.zeros(b, N, len(self.index_map), device=inputs.device)

        # Sum values according to the index map
        for new_idx, old_indices in self.index_map.items():
            for old_idx in old_indices:
                outputs[:, :, new_idx] += inputs[:, :, old_idx]

        # Normalize to get a probability distribution
        sum_outputs = outputs.sum(
            dim=-1, keepdim=True
        )  # Compute the sum across the last dimension
        probabilities = outputs / sum_outputs  # Basic normalization

        # Average across N to get shape [b, 11]
        avg_probabilities = probabilities.mean(dim=1)  # Average over N

        # Compute the cross-entropy loss
        loss = self.criterion(avg_probabilities.log(), targets)

        return loss


class RegularizedDifferentialLoss(nn.Module):
    def __init__(self, reg_lambda=0.1, index_map=index_map):
        super(RegularizedDifferentialLoss, self).__init__()
        self.reg_lambda = reg_lambda
        self.average_ce_loss = AvgCELoss()
        self.differential_loss = GroupedLossWithIndexMap(index_map)

    def forward(self, output, logits, differential):

        cell_classes = logits.argmax(dim=1)

        # Compute the average cross-entropy loss
        ce_loss = self.average_ce_loss(output, cell_classes)

        # Compute the differential loss
        diff_loss = self.differential_loss(output, differential)

        # Combine the losses
        total_loss = diff_loss + self.reg_lambda * ce_loss

        return total_loss
