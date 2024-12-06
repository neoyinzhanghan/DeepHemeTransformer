import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from dataset import TensorStackDataModuleV4

# from torchmetrics import Accuracy, F1Score, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger

# from CELoss import MyCrossEntropyLoss
# from L2Loss import MyL2Loss
from TRL2Loss import MyTRL2Loss
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

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, 2048)

        # Initialize the projection layers to zero
        self.q_proj.weight.data.zero_()
        self.k_proj.weight.data.zero_()
        self.v_proj.weight.data.zero_()
        self.out_proj.weight.data.zero_()

        # Optionally, you can also set the biases to zero if needed
        self.q_proj.bias.data.zero_()
        self.k_proj.bias.data.zero_()
        self.v_proj.bias.data.zero_()
        self.out_proj.bias.data.zero_()

        head_dim = d_model // num_heads

        self.attn = Attn(head_dim=head_dim, use_flash_attention=use_flash_attention)
        self.classifier = nn.Linear(2048, num_classes)

    def baseline_forward(self, logit_stack, non_padding_mask):
        logit_stack_sum = logit_stack.sum(dim=1)
        logit_stack_avg = logit_stack_sum / non_padding_mask.sum(dim=1).unsqueeze(1)

        return logit_stack_avg

    def forward(self, feature_stack, logit_stack, non_padding_mask):

        x = feature_stack

        batch_size, N, d = (
            x.size()
        )  # The shape of x is [batch_size, N, d] where N is the number of tokens and d is the dimension of each token

        assert (
            N == self.num_tokens
        ), f"Number of tokens {N} does not match the model's expectation {self.num_tokens}"

        # class_tokens = self.class_token.expand(batch_size, -1, -1)
        # x = torch.cat([class_tokens, x], dim=1) # TODO remove we no longer need a class token

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

        # right now the attn_output should have the shape [batch_size, N, d_model]
        # apply linear to the last dimension

        attn_output = self.out_proj(attn_output)

        logits_offsets = self.classifier(attn_output)

        # log the logit_stack
        # print(f"logit_stack shape: {logit_stack.shape}")
        logged_logit_stack = torch.log(logit_stack + 1e-8)

        # logits_offsets = F.softmax(logits_offsets, dim=-1)

        # add the logits and the offsets then we need to uniformly subtract 1/num_classes from the logits to make sure the sum is 1
        logits = logged_logit_stack + logits_offsets  # - (1 / self.num_classes)

        logits = F.softmax(logits, dim=-1)

        # print(f"logits shape: {logits.shape}")
        # print(logits[0, 0, :])
        # print(f"Sum of logits: {logits[0, 0, :].sum()}")
        # print(f"logit_stack shape: {logit_stack.shape}")
        # print(logit_stack[0, 0, :])
        # print(f"Sum of logit_stack: {logit_stack[0, 0, :].sum()}")
        # print(f"logits_offsets shape: {logits_offsets.shape}")
        # print(logits_offsets[0, 0, :] - 1 / self.num_classes)
        # print(f"Sum of logits_offsets: {logits_offsets[0, 0, :].sum()}")

        # logits have shape [batch_size, N, num_classes], non_padding_mask has shape [batch_size, N]
        # multiply the logits by the non_padding_mask to zero out the padding tokens
        logits = logits * non_padding_mask.unsqueeze(2)

        # sum the logits across the N dimension
        logits_sum = logits.sum(dim=1)

        # then divide by the sum of the non_padding_mask to get the average
        diff = logits_sum / non_padding_mask.sum(dim=1).unsqueeze(1)

        # print(f"logits shape: {logits.shape}")
        # print(logits[0, :])
        # print(f"logit_stack_avg shape: {logit_stack_avg.shape}")
        # print(logit_stack_avg[0, :])

        return diff


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
        self.loss_fn = MyTRL2Loss()
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

    def forward(self, feature_stack, logit_stack, non_padding_mask):
        logits = self.model(feature_stack, logit_stack, non_padding_mask)
        return logits

    def baseline_forward(self, logit_stack, non_padding_mask):
        logit_stack_avg = self.model.baseline_forward(logit_stack, non_padding_mask)
        return logit_stack_avg

    def training_step(self, batch, batch_idx):
        feature_stack, logit_stack, non_padding_mask, diff_tensor = batch
        logits = self(feature_stack, logit_stack, non_padding_mask)
        y = diff_tensor
        loss = self.loss_fn(y, logits)

        # Custom accuracy metric
        accuracy = self.metric_fn(y, logits)
        a_accuracy = self.a_metric_fn(y, logits)
        r_accuracy = self.r_metric_fn(y, logits)

        # Log training loss and accuracy
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=feature_stack.size(0),
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
        feature_stack, logit_stack, non_padding_mask, diff_tensor = batch
        y = diff_tensor
        logits = self(feature_stack, logit_stack, non_padding_mask)
        loss = self.loss_fn(logits, y)

        # Custom accuracy metric
        accuracy = self.metric_fn(y, logits)
        a_accuracy = self.a_metric_fn(y, logits)
        r_accuracy = self.r_metric_fn(y, logits)

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
        feature_stack, logit_stack, non_padding_mask, diff_tensor = batch
        logits = self(feature_stack, logit_stack, non_padding_mask)
        y = diff_tensor
        loss = self.loss_fn(logits, y)

        # Custom accuracy metric
        accuracy = self.metric_fn(y, logits)
        a_accuracy = self.a_metric_fn(y, logits)
        r_accuracy = self.r_metric_fn(y, logits)

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
    logit_stacks_dir,
    diff_data_path,
    num_gpus=2,
    num_epochs=100,
    batch_size=16,
    lr=0.0005,
    num_workers=8,
    num_heads=1,
    num_classes=9,
    use_flash_attention=True,
    log_dir="lightning_logs_V4",
    experiment_name="multihead_attention_classifier",
    message="No message",
):
    data_module = TensorStackDataModuleV4(
        feature_stacks_dir=feature_stacks_dir,
        logit_stacks_dir=logit_stacks_dir,
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
        # strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module.train_dataloader())


if __name__ == "__main__":
    feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
    logit_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"
    diff_data_path = (
        "/media/hdd3/neo/DiffTransformerV1DataMini/subsampled_split_diff_data.csv"
    )

    message = "Testing different learning rates for the simple transformer model using the full mini dataset with the random subsample data augmentation, using simple L2 loss and the AR_acc metric."

    for lr in [0.1]:  # 0.00000001]:

        train_model(
            feature_stacks_dir=feature_stacks_dir,
            logit_stacks_dir=logit_stacks_dir,
            diff_data_path=diff_data_path,
            num_gpus=2,
            num_epochs=50,
            batch_size=5,  # 16,
            lr=lr,
            num_heads=8,
            num_classes=9,
            use_flash_attention=True,
            message=message,
        )
