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

        return x


class DeepHemeTransformer(nn.Module):
    def __init__(self):
        super(DeepHemeTransformer, self).__init__()

        self.feature_projector = nn.Linear(2048, 1024)
        self.transformer = MultiHeadAttentionClassifier(d_model=1024, num_heads=8)
        self.last_layer_linear = nn.Linear(1024, 23)

    def forward(self, x):

        # x should be shaped like [b, N, 2048]
        batch_size = x.size(0)
        num_cells = x.size(1)

        # reshape x to [b * N, 2048]
        x = x.view(-1, 2048)
        assert (
            x.size(0) == batch_size * num_cells
        ), f"Checkpoint 0: x.size(0)={x.size(0)}, expected {batch_size * num_cells}"

        # project features to 1024
        x = self.feature_projector(x)
        assert x.size(1) == 1024, f"Checkpoint 1: x.size(1)={x.size(1)}, expected 1024"
        assert (
            x.size(0) == batch_size * num_cells
        ), f"Checkpoint 2: x.size(0)={x.size(0)}, expected {batch_size * num_cells}"

        # reshape x to [b, N, 1024]
        x = x.view(batch_size, num_cells, 1024)
        assert (
            x.size(0) == batch_size
        ), f"Checkpoint 3: x.size(0)={x.size(0)}, expected {batch_size}"
        assert (
            x.size(1) == num_cells
        ), f"Checkpoint 4: x.size(1)={x.size(1)}, expected {num_cells}"

        # pass through transformer
        x = self.transformer(x)
        assert (
            x.size(0) == batch_size
        ), f"Checkpoint 5: x.size(0)={x.size(0)}, expected {batch_size}"
        assert (
            x.size(1) == num_cells
        ), f"Checkpoint 6: x.size(1)={x.size(1)}, expected {num_cells}"
        assert x.size(2) == 1024, f"Checkpoint 7: x.size(2)={x.size(2)}, expected 1024"

        # reshape x to [b * N, 1024]
        x = x.view(-1, 1024)
        assert (
            x.size(0) == batch_size * num_cells
        ), f"Checkpoint 8: x.size(0)={x.size(0)}, expected {batch_size * num_cells}"

        # pass through final linear layer
        x = self.last_layer_linear(x)
        assert (
            x.size(0) == batch_size * num_cells
        ), f"Checkpoint 9: x.size(0)={x.size(0)}, expected {batch_size * num_cells}"

        # reshape x to [b, N, 23]
        x = x.view(batch_size, num_cells, 23)
        assert (
            x.size(0) == batch_size
        ), f"Checkpoint 10: x.size(0)={x.size(0)}, expected {batch_size}"

        return x


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

# Assuming DeepHemeTransformer and CellFeaturesDataModule are implemented elsewhere
from your_model_module import DeepHemeTransformer
from cell_dataloader import CellFeaturesDataModule


class DeepHemeModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, max_epochs=50, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepHemeTransformer()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, logits, differential = batch
        outputs = self(features)

        # Reshape the outputs and differential to [batch_size * num_cells, 23]
        outputs = outputs.view(-1, 23)
        differential = differential.view(-1)

        loss = self.loss_fn(outputs, differential)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        features, logits, differential = batch
        outputs = self(features)

        # Reshape the outputs and differential to [batch_size * num_cells, 23]
        outputs = outputs.view(-1, 23)
        differential = differential.view(-1)

        loss = self.loss_fn(outputs, differential)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

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
    # Instantiate the DataModule
    metadata_file_path = "/media/hdd3/neo/DeepHemeTransformerData/features_metadata.csv"
    batch_size = 32

    datamodule = CellFeaturesDataModule(
        metadata_file=metadata_file_path, batch_size=batch_size
    )

    # Define a PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=50,
        log_every_n_steps=10,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,
    )

    # Create an instance of your LightningModule
    model = DeepHemeModule(learning_rate=1e-3, max_epochs=50, weight_decay=1e-2)

    # Train the model
    trainer.fit(model, datamodule)
