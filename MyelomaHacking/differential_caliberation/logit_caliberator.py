import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LinearLR
from TRL2Loss import MyTRL2Loss
from AR_acc import AR_acc, A_acc, R_acc, Class_AR_acc, Class_A_acc, Class_R_acc
from BMAassumptions import BMA_final_classes
from CDdataset import TensorStackDataModuleV5


class LogitCalibrator(pl.LightningModule):
    def __init__(self, offset_tensor, d, D):
        """
        Initialize the LogitCalibrator module.

        Args:
            offset_tensor (torch.Tensor): A tensor of shape [9] to be added as an offset to logits.
            d (float): Parameter for AR_acc.
            D (float): Parameter for AR_acc.
        """
        super(LogitCalibrator, self).__init__()
        if offset_tensor.shape != (9,):
            raise ValueError("offset_tensor must have shape [9].")
        self.offset_tensor = nn.Parameter(offset_tensor, requires_grad=True)
        self.criterion = MyTRL2Loss()

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

    def forward(self, logits_stack):
        """
        Forward pass for the LogitCalibrator module.

        Args:
            logits_stack (torch.Tensor): A tensor of shape [b, N, 9] containing logits.

        Returns:
            torch.Tensor: A tensor of shape [b, 9] representing the calibrated logits.
        """
        if logits_stack.ndim != 3 or logits_stack.shape[-1] != 9:
            raise ValueError("logits_stack must have shape [b, N, 9].")

        # Take the log of the logits with a small constant added to avoid log(0)
        logits_stack_logged = torch.log(logits_stack + 1)

        # Add the offset tensor along the last dimension
        logits_stack_logged_offset = logits_stack_logged + self.offset_tensor

        # Apply softmax across the last dimension
        logits_stack_logged_offset_softmax = F.softmax(
            logits_stack_logged_offset, dim=-1
        )

        # Average the softmaxed logits across the N dimension
        logits_stack_logged_offset_softmax_avg = torch.mean(
            logits_stack_logged_offset_softmax, dim=1
        )

        return logits_stack_logged_offset_softmax_avg

    def training_step(self, batch, batch_idx):
        """
        Training step for the LogitCalibrator module.

        Args:
            batch (tuple): A tuple containing the input batch.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        _, logit_stack, _, diff_tensor, _ = batch
        logits_calibrated = self.forward(logit_stack)
        loss = self.criterion(logits_calibrated, diff_tensor)
        self.log("train_loss", loss)

        # Log metrics
        acc = self.metric_fn(logits_calibrated, diff_tensor, self.d, self.D)
        a_acc = self.a_metric_fn(logits_calibrated, diff_tensor)
        r_acc = self.r_metric_fn(logits_calibrated, diff_tensor)
        self.log("train_AR_acc", acc)
        self.log("train_A_acc", a_acc)
        self.log("train_R_acc", r_acc)

        for bma_class in BMA_final_classes:
            class_acc = self.class_metric_fns[bma_class](logits_calibrated, diff_tensor)
            a_class_acc = self.a_class_metric_fns[bma_class](
                logits_calibrated, diff_tensor
            )
            r_class_acc = self.r_class_metric_fns[bma_class](
                logits_calibrated, diff_tensor
            )
            self.log(f"train_AR_acc_{bma_class}", class_acc)
            self.log(f"train_A_acc_{bma_class}", a_class_acc)
            self.log(f"train_R_acc_{bma_class}", r_class_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the LogitCalibrator module.

        Args:
            batch (tuple): A tuple containing the input batch.
            batch_idx (int): Index of the current batch.

        Returns:
            None
        """
        _, logit_stack, _, diff_tensor, _ = batch
        logits_calibrated = self.forward(logit_stack)
        loss = self.criterion(logits_calibrated, diff_tensor)
        self.log("val_loss", loss, prog_bar=True)

        acc = self.metric_fn(logits_calibrated, diff_tensor, self.d, self.D)
        a_acc = self.a_metric_fn(logits_calibrated, diff_tensor)
        r_acc = self.r_metric_fn(logits_calibrated, diff_tensor)
        self.log("val_AR_acc", acc, prog_bar=True)
        self.log("val_A_acc", a_acc, prog_bar=True)
        self.log("val_R_acc", r_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step for the LogitCalibrator module.

        Args:
            batch (tuple): A tuple containing the input batch.
            batch_idx (int): Index of the current batch.

        Returns:
            None
        """
        _, logit_stack, _, diff_tensor, _ = batch
        logits_calibrated = self.forward(logit_stack)
        loss = self.criterion(logits_calibrated, diff_tensor)
        self.log("test_loss", loss)

        acc = self.metric_fn(logits_calibrated, diff_tensor, self.d, self.D)
        a_acc = self.a_metric_fn(logits_calibrated, diff_tensor)
        r_acc = self.r_metric_fn(logits_calibrated, diff_tensor)
        self.log("test_AR_acc", acc)
        self.log("test_A_acc", a_acc)
        self.log("test_R_acc", r_acc)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_model(self, data_module, max_epochs=10, gpus=1):
        """
        Train the model using the provided data module.

        Args:
            data_module (LightningDataModule): The data module containing train, validation, and test datasets.
            max_epochs (int): The number of epochs to train for.
            gpus (int): The number of GPUs to use for training.

        Returns:
            None
        """
        trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
        trainer.fit(self, datamodule=data_module)
        trainer.test(self, datamodule=data_module)


if __name__ == "__main__":
    # Directory paths
    feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
    logit_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"
    diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"
    dx_data_path = "/media/hdd3/neo/dx_data_test.csv"
    batch_size = 32  # Adjust as needed
    num_workers = 4  # Adjust as needed

    # Initialize Data Module
    data_module = TensorStackDataModuleV5(
        feature_stacks_dir=feature_stacks_dir,
        logit_stacks_dir=logit_stacks_dir,
        diff_data_path=diff_data_path,
        dx_data_path=dx_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Create model instance
    offset_tensor = torch.zeros(9)  # Replace with appropriate initialization
    d = 1.0  # Adjust based on your AR_acc parameters
    D = 1.0  # Adjust based on your AR_acc parameters
    model = LogitCalibrator(offset_tensor, d, D)

    # Train the model
    model.train_model(data_module, max_epochs=10, gpus=1)
