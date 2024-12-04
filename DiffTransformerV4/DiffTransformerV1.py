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

# from TRL2Loss import MyTRL2Loss

# from TRL2Loss import MyTRL2Loss
from AR_acc import AR_acc, A_acc, R_acc, Class_AR_acc, Class_A_acc, Class_R_acc
from BMAassumptions import BMA_final_classes


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
        num_tokens=3000,
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

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
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
        d=0.2,
        D=0.02,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiHeadAttentionClassifier(
            d_model=d_model,
            num_heads=num_heads,
            num_classes=num_classes,
            use_flash_attention=use_flash_attention,
        )

        # Custom loss and accuracy metric
        self.loss_fn = MyL2Loss()
        self.metric_fn = AR_acc()
        self.a_metric_fn = A_acc()
        self.r_metric_fn = R_acc()

        self.class_metric_fns = {}
        self.a_class_metric_fns = {}
        self.r_class_metric_fns = {}

        for bma_class in BMA_final_classes:
            self.class_metric_fns[bma_class] = Class_AR_acc(class_name=bma_class)
            self.a_class_metric_fns[bma_class] = Class_A_acc(class_name=bma_class)
            self.r_class_metric_fns[bma_class] = Class_R_acc(class_name=bma_class)

        # Parameters for AR_acc
        self.d = d
        self.D = D

    def forward(self, x):
        logits = self.model(x)
        return F.softmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(y, logits)

        # Custom accuracy metric
        accuracy = self.metric_fn(y, logits, d=self.d, D=self.D)
        a_accuracy = self.a_metric_fn(y, logits, d=self.d, D=self.D)
        r_accuracy = self.r_metric_fn(y, logits, d=self.d, D=self.D)

        # Log training loss and accuracy
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=x.size(0),
        )
        self.log("train_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "train_a_accuracy", a_accuracy, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_r_accuracy", r_accuracy, on_epoch=True, prog_bar=True, logger=True
        )

        for bma_class in BMA_final_classes:
            class_accuracy = self.class_metric_fns[bma_class](y, logits)
            self.log(
                f"train_{bma_class}_accuracy",
                class_accuracy,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        for bma_class in BMA_final_classes:
            a_class_accuracy = self.a_class_metric_fns[bma_class](y, logits)
            self.log(
                f"train_{bma_class}_a_accuracy",
                a_class_accuracy,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        for bma_class in BMA_final_classes:
            r_class_accuracy = self.r_class_metric_fns[bma_class](y, logits)
            self.log(
                f"train_{bma_class}_r_accuracy",
                r_class_accuracy,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Custom accuracy metric
        accuracy = self.metric_fn(y, logits, d=self.d, D=self.D)
        a_accuracy = self.a_metric_fn(y, logits, d=self.d, D=self.D)
        r_accuracy = self.r_metric_fn(y, logits, d=self.d, D=self.D)

        # Log validation loss and accuracy
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val_a_accuracy", a_accuracy, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_r_accuracy", r_accuracy, on_epoch=True, prog_bar=True, logger=True
        )

        for bma_class in BMA_final_classes:
            class_accuracy = self.class_metric_fns[bma_class](y, logits)
            self.log(
                f"val_{bma_class}_accuracy",
                class_accuracy,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        for bma_class in BMA_final_classes:
            a_class_accuracy = self.a_class_metric_fns[bma_class](y, logits)
            self.log(
                f"val_{bma_class}_a_accuracy",
                a_class_accuracy,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        for bma_class in BMA_final_classes:
            r_class_accuracy = self.r_class_metric_fns[bma_class](y, logits)
            self.log(
                f"val_{bma_class}_r_accuracy",
                r_class_accuracy,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Custom accuracy metric
        accuracy = self.metric_fn(y, logits, d=self.d, D=self.D)
        a_accuracy = self.a_metric_fn(y, logits, d=self.d, D=self.D)
        r_accuracy = self.r_metric_fn(y, logits, d=self.d, D=self.D)

        # Log test loss and accuracy
        self.log("test_loss", loss, on_epoch=True, logger=True)
        self.log("test_accuracy", accuracy, on_epoch=True, logger=True)
        self.log("test_a_accuracy", a_accuracy, on_epoch=True, logger=True)
        self.log("test_r_accuracy", r_accuracy, on_epoch=True, logger=True)

        for bma_class in BMA_final_classes:
            class_accuracy = self.class_metric_fns[bma_class](y, logits)
            self.log(
                f"test_{bma_class}_accuracy",
                class_accuracy,
                on_epoch=True,
                logger=True,
            )

        for bma_class in BMA_final_classes:
            a_class_accuracy = self.a_class_metric_fns[bma_class](y, logits)
            self.log(
                f"test_{bma_class}_a_accuracy",
                a_class_accuracy,
                on_epoch=True,
                logger=True,
            )

        for bma_class in BMA_final_classes:
            r_class_accuracy = self.r_class_metric_fns[bma_class](y, logits)
            self.log(
                f"test_{bma_class}_r_accuracy",
                r_class_accuracy,
                on_epoch=True,
                logger=True,
            )

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
            self.log("lr", current_lr, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.num_epochs, eta_min=0
        )
        return [optimizer], [scheduler]


def train_model(
    feature_stacks_dir,
    diff_data_path,
    num_gpus=2,
    num_epochs=100,
    batch_size=16,
    lr=0.0005,
    num_workers=8,
    num_heads=1,
    num_classes=9,
    use_flash_attention=True,
    log_dir="lightning_logs",
    experiment_name="multihead_attention_classifier",
    message="No message",
):
    data_module = TensorStackDataModule(
        feature_stacks_dir=feature_stacks_dir,
        diff_data_path=diff_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = MultiHeadAttentionClassifierPL(
        d_model=2048,
        num_heads=num_heads,
        num_classes=num_classes,
        use_flash_attention=use_flash_attention,
        num_epochs=num_epochs,
        lr=lr,
    )
    logger = TensorBoardLogger(log_dir, name=experiment_name)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=num_gpus,
        accelerator="gpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module.train_dataloader())

    # save the message as txt file in the experiment directory
    with open(os.path.join(logger.log_dir, "message.txt"), "w") as f:
        f.write(message)


if __name__ == "__main__":
    feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
    diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"

    message = "Testing different learning rates for the simple transformer model using the full mini dataset with the random subsample data augmentation, using simple L2 loss and the AR_acc metric."

    for lr in [0.00005]:
        train_model(
            feature_stacks_dir,
            diff_data_path,
            num_gpus=2,
            num_epochs=100,
            batch_size=16,  # 16,
            lr=lr,
            num_heads=1,
            num_classes=9,
            use_flash_attention=True,
            message=message,
        )
