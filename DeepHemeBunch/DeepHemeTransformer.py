###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
# Base Pytorch Model Definition
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_fn import RegularizedDifferentialLoss
from cell_dataloader import CellFeaturesLogitsDataset, CellFeaturesDataModule
from loss_fn import RegularizedDifferentialLoss
from BMAassumptions import index_map


class DeepHemeTransformer(nn.Module):
    def __init__(self, num_heads=8):
        super(DeepHemeTransformer, self).__init__()
        self.last_layer_linear = nn.Linear(2048, 23)

    def forward(self, x):

        x_ele = x
        # x should be a tensor of shape [N, 2048] where N is the number of cells

        num_cells = x_ele.size(0)

        # pass through final linear layer
        x_ele = self.last_layer_linear(x_ele)
        assert (
            x_ele.size(0) == num_cells
        ), f"Checkpoint 9: x_ele.size(0)={x_ele.size(0)}, expected {batch_size * num_cells}"

        # assert that the output shape is [N, 23]
        assert (
            x_ele.size(1) == 23
        ), f"Checkpoint 10: x_ele.size(1)={x_ele.size(1)}, expected 23"

        return x_ele

    def predict_diff(self, x, index_map=index_map):
        x_ele = self.forward(x)

        # Initialize an output tensor for the summed values
        N = x_ele.shape[0]

        outputs = torch.zeros(N, len(index_map), device=x_ele.device)

        # Sum values according to the index map
        for new_idx, old_indices in self.index_map.items():
            for old_idx in old_indices:
                outputs[:, new_idx] += x_ele[:, old_idx]

        # Normalize to get a probability distribution
        sum_outputs = outputs.sum(
            dim=-1, keepdim=True
        )  # Compute the sum across the last dimension
        probabilities = outputs / sum_outputs  # Basic normalization

        return probabilities


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
# Pytorch Lightning Model Definition & Training Script
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

# Assuming DeepHemeTransformer and CellFeaturesDataModule are implemented elsewhere
from cell_dataloader import CellFeaturesDataModule


class DeepHemeModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        max_epochs=50,
        weight_decay=1e-2,
        num_heads=8,
        reg_lambda1=0,
        reg_lambda2=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepHemeTransformer(num_heads=num_heads)
        self.loss_fn = RegularizedDifferentialLoss(
            reg_lambda1=reg_lambda1, reg_lambda2=reg_lambda2
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features_list, logits_list, differential_list = batch
        outputs_list = []

        # Iterate over the list of inputs in the batch
        for features, logits, differential in zip(
            features_list, logits_list, differential_list
        ):
            outputs = self(features)
            outputs_list.append(outputs)

        loss = self.loss_fn(outputs_list, logits_list, differential_list)

        # Log and return a dictionary with relevant metrics
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(features_list),
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        features_list, logits_list, differential_list = batch
        outputs_list = []

        # Iterate over the list of inputs in the batch
        for features, logits, differential in zip(
            features_list, logits_list, differential_list
        ):
            outputs = self(features)
            outputs_list.append(outputs)

        # Compute the loss for the batch
        loss = self.loss_fn(outputs_list, logits_list, differential_list)

        # Log the validation loss and return it for tracking
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(features_list),
        )
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "cosine_decay",
            },
        }

    def on_train_epoch_start(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True, logger=True)


def load_model(checkpoint_path):
    """
    Load a DeepHemeModule model from a specified checkpoint.

    Args:
        checkpoint_path (str): Path to the saved PyTorch Lightning checkpoint.

    Returns:
        DeepHemeModule: The model loaded from the checkpoint.
    """
    # Load the model from checkpoint
    model = DeepHemeModule.load_from_checkpoint(checkpoint_path)
    return model


if __name__ == "__main__":

    import numpy as np

    for i in range(1):
        # learning_rate = 10 ** np.random.uniform(-10, 0)
        # Set up parameters
        metadata_file_path = (
            "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
        )
        batch_size = 32
        # Instantiate the DataModule
        datamodule = CellFeaturesDataModule(
            metadata_file=metadata_file_path, batch_size=batch_size
        )

        # use a 1e-4 learning rate
        learning_rate = 1e-4

        # Set up the logger with a subfolder named after the learning rate
        log_dir = f"logs/train_nov3/lr_1e-4_no_reg"
        logger = TensorBoardLogger(
            save_dir=log_dir,
            name="",
        )

        # Define a PyTorch Lightning trainer with the custom logger
        trainer = pl.Trainer(
            max_epochs=50,
            log_every_n_steps=10,
            devices=1,
            accelerator="gpu",
            logger=logger,
        )

        # Create an instance of your LightningModule
        model = DeepHemeModule(
            learning_rate=learning_rate, max_epochs=50, weight_decay=1e-2
        )

        # Train the model
        trainer.fit(model, datamodule)
