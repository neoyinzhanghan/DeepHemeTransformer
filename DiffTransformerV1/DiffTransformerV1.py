import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from dataset import TensorStackDataModule

# from torchmetrics import Accuracy, F1Score, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger

# from CELoss import MyCrossEntropyLoss
from L2Loss import MyL2Loss


class Attn(nn.Module):
    def __init__(self, head_dim, use_flash_attention):
        super(Attn, self).__init__()
        self.head_dim = head_dim
        self.use_flash_attention = use_flash_attention

    def forward(self, q, k, v):
        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)
        return attn_output


class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self,
        d_model=2048,
        num_heads=8,
        num_classes=9,
        num_tokens=100,
        use_flash_attention=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash_attention = use_flash_attention
        self.num_classes = num_classes
        self.num_tokens = num_tokens

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        head_dim = d_model // num_heads

        self.attn = Attn(head_dim=head_dim, use_flash_attention=use_flash_attention)

        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, N, d = (
            x.size()
        )  # The shape of x is [batch_size, N, d] where N is the number of tokens and d is the dimension of each token

        assert (
            N == self.num_tokens
        ), f"Number of tokens {N} does not match the model's expectation {self.num_tokens}"

        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

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

        attn_output = self.attn(q, k, v)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.out_proj(attn_output)

        class_token_output = output[:, 0]
        logits = self.classifier(class_token_output)
        return logits


class MultiHeadAttentionClassifierPL(pl.LightningModule):
    def __init__(
        self,
        d_model,
        num_heads,
        num_classes,
        use_flash_attention=True,
        num_epochs=50,
        lr=0.0005,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiHeadAttentionClassifier(
            d_model=d_model,
            num_heads=num_heads,
            num_classes=num_classes,
            use_flash_attention=use_flash_attention,
        )

        # self.train_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        # self.val_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        # self.test_accuracy = Accuracy(num_classes=num_classes, task="multiclass")

        # self.train_f1 = F1Score(num_classes=num_classes, task="multiclass")
        # self.val_f1 = F1Score(num_classes=num_classes, task="multiclass")
        # self.test_f1 = F1Score(num_classes=num_classes, task="multiclass")

        # self.train_auroc = AUROC(num_classes=num_classes, task="multiclass")
        # self.val_auroc = AUROC(num_classes=num_classes, task="multiclass")
        # self.test_auroc = AUROC(num_classes=num_classes, task="multiclass")

        self.loss_fn = MyL2Loss()

    def forward(self, x):
        logits = self.model(x)  # should have shape [batch_size, num_classes]

        # apply a softmax operation to the logits to get probabilities
        return F.softmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # print the shape of x and y
        # print(x.shape, y.shape)

        logits = self(x)

        # print(logits.shape, y.shape)
        # import sys
        # sys.exit()
        loss = self.loss_fn(logits, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,  # Log at each step
            on_epoch=True,  # Log at the end of each epoch
            prog_bar=True,  # Display in the progress bar
            logger=True,  # Send to the logger (e.g., TensorBoard)
            batch_size=x.size(0),
        )
        # self.log("train_accuracy", self.train_accuracy(logits, y))
        # self.log("train_f1", self.train_f1(logits, y))
        # self.log("train_auroc", self.train_auroc(logits, y))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        # self.log("val_accuracy", self.val_accuracy(logits, y))
        # self.log("val_f1", self.val_f1(logits, y))
        # self.log("val_auroc", self.val_auroc(logits, y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)
        # self.log("test_accuracy", self.test_accuracy(logits, y))
        # self.log("test_f1", self.test_f1(logits, y))
        # self.log("test_auroc", self.test_auroc(logits, y))

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]
        self.log("lr", current_lr)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.num_epochs, eta_min=0
        )
        return [optimizer], [scheduler]


def train_model(
    feature_stacks_dir,
    diff_data_path,
    num_heads=1,
    d_model=2048,
    num_classes=9,
    batch_size=16,
    num_workers=8,
    num_gpus=2,
    num_epochs=50,
    lr=0.0005,
):
    data_module = TensorStackDataModule(
        feature_stacks_dir=feature_stacks_dir,
        diff_data_path=diff_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = MultiHeadAttentionClassifierPL(
        d_model=d_model,
        num_heads=num_heads,
        num_classes=num_classes,
        use_flash_attention=True,
        num_epochs=num_epochs,
        lr=lr,
    )

    logger = TensorBoardLogger("lightning_logs", name="multihead_attention")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=num_gpus,
        accelerator="gpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module.val_dataloader())


if __name__ == "__main__":
    feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
    diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"

    for lr in [
        0.00005
    ]:  # [5, 0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005, 0.00000005]:
        train_model(
            feature_stacks_dir=feature_stacks_dir,
            diff_data_path=diff_data_path,
            batch_size=5,
            num_gpus=2,
            num_epochs=50,
            lr=lr,
        )
