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

# usage example for the loss loss(forward_output, logits, differential)
# the dataset output is a tuple of features, logits, differential


class Attn(nn.Module):
    def __init__(self, head_dim, use_flash_attention):
        super(Attn, self).__init__()
        self.head_dim = head_dim
        self.use_flash_attention = use_flash_attention

    def forward(self, q, k, v):
        if self.use_flash_attention:
            # Use PyTorch's built-in scaled dot product attention with flash attention support
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Compute scaled dot product attention manually
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)
        return attn_output


class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self,
        d_model=1024,
        num_heads=8,
        use_flash_attention=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash_attention = use_flash_attention

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection after attention
        self.out_proj = nn.Linear(d_model, d_model)

        # The attention mechanism
        self.attn = Attn(
            head_dim=self.head_dim, use_flash_attention=use_flash_attention
        )

    def forward(self, x):
        # Shape of x: (batch_size, N, d_model), where N is the sequence length

        batch_size = x.size(0)

        input_shape = x.size()
        # Linear projections for Q, K, V (batch_size, num_heads, N+1, head_dim)
        q = (
            self.q_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply attention (batch_size, num_heads, N+1, head_dim)
        attn_output = self.attn(q, k, v)

        # Concatenate attention output across heads (batch_size, N+1, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # Apply final linear projection
        x = self.out_proj(attn_output)

        # check that the output and input shapes match
        output_shape = x.size()

        assert (
            output_shape == input_shape
        ), f"Output shape {output_shape} does not match input shape {input_shape} in transformer forward."

        return x


class DeepHemeTransformer(nn.Module):
    def __init__(self, num_heads=8):
        super(DeepHemeTransformer, self).__init__()

        self.feature_projector = nn.Linear(2048, 1024)
        self.transformer = MultiHeadAttentionClassifier(
            d_model=1024, num_heads=num_heads
        )
        self.last_layer_linear = nn.Linear(1024, 23)

    def forward(self, x):

        x_ele = x
        # x should be a list of inputs with shape [N, 2048]

        num_cells = x_ele.size(0)

        # project features to 1024
        x_ele = self.feature_projector(x_ele)
        assert (
            x_ele.size(1) == 1024
        ), f"Checkpoint 1: x_ele.size(1)={x_ele.size(1)}, expected 1024"
        assert (
            x_ele.size(0) == num_cells
        ), f"Checkpoint 2: x_ele.size(0)={x_ele.size(0)}, expected {num_cells}"

        # add a batch dimension to x_ele
        x_ele = x_ele.unsqueeze(0)
        assert (
            x_ele.size(0) == 1
        ), f"Checkpoint 3: x_ele.size(0)={x_ele.size(0)}, expected 1"
        assert (
            x_ele.size(1) == num_cells
        ), f"Checkpoint 4: x_ele.size(1)={x_ele.size(1)}, expected {num_cells}"
        assert (
            x_ele.size(2) == 1024
        ), f"Checkpoint 5: x_ele.size(2)={x_ele.size(2)}, expected 1024"

        # pass through transformer
        x_ele = self.transformer(x_ele)
        assert (
            x_ele.size(0) == 1
        ), f"Checkpoint 6: x_ele.size(0)={x.size(0)}, expected {1}"
        assert (
            x_ele.size(1) == num_cells
        ), f"Checkpoint 7: x_ele.size(1)={x_ele.size(1)}, expected {num_cells}"
        assert (
            x_ele.size(2) == 1024
        ), f"Checkpoint 8: x_ele.size(2)={x_ele.size(2)}, expected 1024"

        # reshape x to [N, 1024] by removing the batch dimension
        x_ele = x_ele.squeeze(0)

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
        reg_lambda=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepHemeTransformer(num_heads=num_heads)
        self.loss_fn = RegularizedDifferentialLoss(reg_lambda=reg_lambda)

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
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
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


if __name__ == "__main__":

    import numpy as np

    for i in range(10):
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
        log_dir = f"logs/lr_1e-4_no_reg"
        logger = TensorBoardLogger(
            save_dir=log_dir,
            name="",
        )

        # Define a PyTorch Lightning trainer with the custom logger
        trainer = pl.Trainer(
            max_epochs=20,
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
