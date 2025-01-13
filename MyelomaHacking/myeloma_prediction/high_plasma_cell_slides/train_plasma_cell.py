import os
import torch
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import albumentations as A
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms, datasets, models
from torchmetrics import Accuracy, AUROC
from torch.utils.data import WeightedRandomSampler
from dataset import CustomPlasmaCellDataset, grouped_label_to_index

############################################################################
####### DEFINE HYPERPARAMETERS AND DATA DIRECTORIES ########################
############################################################################

num_epochs = 50
default_config = {"lr": 3.56e-05}  # 1.462801279401232e-06}
base_data_dir = "/media/hdd3/neo/pooled_deepheme_data"
num_gpus = 2
num_workers = 64
downsample_factor = 1
batch_size = 256
img_size = 96
num_classes = 11

############################################################################
####### FUNCTIONS FOR DATA AUGMENTATION AND DATA LOADING ###################
############################################################################


def get_feat_extract_augmentation_pipeline(image_size):
    """Returns a randomly chosen augmentation pipeline for SSL."""

    ## Simple augmentation to improve the data generalizability
    transform_shape = A.Compose(
        [
            A.ShiftScaleRotate(p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(shear=(-10, 10), p=0.3),
            A.ISONoise(
                color_shift=(0.01, 0.02),
                intensity=(0.05, 0.01),
                always_apply=False,
                p=0.2,
            ),
        ]
    )
    transform_color = A.Compose(
        [
            A.RandomBrightnessContrast(contrast_limit=0.4, brightness_limit=0.4, p=0.5),
            A.CLAHE(p=0.3),
            A.ColorJitter(p=0.2),
            A.RandomGamma(p=0.2),
        ]
    )

    # Compose the two augmentation pipelines
    return A.Compose(
        [A.Resize(image_size, image_size), A.OneOf([transform_shape, transform_color])]
    )


# Define a custom dataset that applies downsampling
class DownsampledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, downsample_factor, apply_augmentation=True):
        self.dataset = dataset
        self.downsample_factor = downsample_factor
        self.apply_augmentation = apply_augmentation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.downsample_factor > 1:
            size = (96 // self.downsample_factor, 96 // self.downsample_factor)
            image = transforms.functional.resize(image, size)

        # Convert image to RGB if not already
        image = to_pil_image(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.apply_augmentation:
            # Apply augmentation
            image = get_feat_extract_augmentation_pipeline(
                image_size=96 // self.downsample_factor
            )(image=np.array(image))["image"]

        image = to_tensor(image)

        return image, label


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        combined_metadata_csv_path,
        batch_size,
        downsample_factor,
    ):
        super().__init__()
        self.combined_metadata_csv_path = combined_metadata_csv_path
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor

    def setup(self, stage=None):
        # Load base train, validation, and test datasets
        train_dataset = CustomPlasmaCellDataset(
            combined_metadata_csv_path=self.combined_metadata_csv_path,
            split="train",
        )
        val_dataset = CustomPlasmaCellDataset(
            combined_metadata_csv_path=self.combined_metadata_csv_path,
            split="val",
        )
        test_dataset = CustomPlasmaCellDataset(
            combined_metadata_csv_path=self.combined_metadata_csv_path,
            split="test",
        )

        self.train_dataset = DownsampledDataset(
            train_dataset, self.downsample_factor, apply_augmentation=True
        )

        self.val_dataset = DownsampledDataset(
            val_dataset, self.downsample_factor, apply_augmentation=False
        )
        self.test_dataset = DownsampledDataset(
            test_dataset, self.downsample_factor, apply_augmentation=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=24,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=24,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=24,
        )


grouped_label_to_index = {
    "blasts and blast-equivalents": 0,
    "promyelocytes": 1,
    "myelocytes": 2,
    "metamyelocytes": 3,
    "neutrophils/bands": 4,
    "monocytes": 5,
    "eosinophils": 6,
    "erythroid precursors": 7,
    "lymphocytes": 8,
    "plasma cells": 9,
    "skippocytes": 10,
}


# Model Module
class Myresnext50(pl.LightningModule):
    def __init__(self, num_classes=11, config=default_config):
        super(Myresnext50, self).__init__()
        self.pretrained = models.resnext50_32x4d(pretrained=True)
        self.pretrained.fc = nn.Linear(self.pretrained.fc.in_features, num_classes)
        # self.my_new_layers = nn.Sequential(
        #     nn.Linear(
        #         1000, 100
        #     ),  # Assuming the output of your pre-trained model is 1000
        #     nn.ReLU(),
        #     nn.Linear(100, num_classes),
        # )
        # self.num_classes = num_classes

        task = "multiclass"

        self.train_accuracy = Accuracy(task=task, num_classes=num_classes).to(
            self.device
        )
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes).to(self.device)
        self.train_auroc = AUROC(num_classes=num_classes, task=task).to(self.device)
        self.val_auroc = AUROC(num_classes=num_classes, task=task).to(self.device)
        self.test_accuracy = Accuracy(num_classes=num_classes, task=task).to(
            self.device
        )
        self.test_auroc = AUROC(num_classes=num_classes, task=task).to(self.device)

        # Per-class accuracy metrics using multiclass task
        self.train_class_accuracies = {}
        self.val_class_accuracies = {}
        self.test_class_accuracies = {}
        for name in grouped_label_to_index.keys():
            setattr(
                self,
                f"train_acc_{name}",
                Accuracy(task=task, num_classes=num_classes).to(self.device),
            )
            setattr(
                self,
                f"val_acc_{name}",
                Accuracy(task=task, num_classes=num_classes).to(self.device),
            )
            setattr(
                self,
                f"test_acc_{name}",
                Accuracy(task=task, num_classes=num_classes).to(self.device),
            )

        self.config = config

    def forward(self, x):
        x = self.pretrained(x)
        return x

    def extract_features(self, x):
        # Extract features before the last fc layer
        layers = list(self.pretrained.children())[:-1]  # Remove the last fc layer
        feature_extractor = nn.Sequential(*layers)
        x = feature_extractor(x)
        x = nn.Flatten()(x)  # Flatten the output if needed
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.train_accuracy(y_hat, y)
        self.train_auroc(y_hat, y)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=True)

        # Per-class accuracy logging
        preds = torch.argmax(y_hat, dim=1)
        for class_name, class_idx in grouped_label_to_index.items():
            class_mask = y == class_idx
            if class_mask.sum() > 0:  # Only compute if we have samples of this class
                y_hat_class = y_hat[class_mask].to(self.device)
                y_class = y[class_mask].to(self.device)
                metric = getattr(self, f"train_acc_{class_name}")
                metric(y_hat_class, y_class)
                self.log(
                    f"train_acc_{class_name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"train_acc_{class_name}",
                )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_accuracy(y_hat, y)
        self.val_auroc(y_hat, y)

        # Per-class accuracy logging
        preds = torch.argmax(y_hat, dim=1)
        for class_name, class_idx in grouped_label_to_index.items():
            class_mask = y == class_idx
            if class_mask.sum() > 0:
                y_hat_class = y_hat[class_mask].to(self.device)
                y_class = y[class_mask].to(self.device)
                metric = getattr(self, f"val_acc_{class_name}")
                metric(y_hat_class, y_class)
                self.log(
                    f"val_acc_{class_name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"val_acc_{class_name}",
                )

        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_accuracy.compute())
        self.log("val_auroc_epoch", self.val_auroc.compute())
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_accuracy(y_hat, y)
        self.test_auroc(y_hat, y)

        # Per-class accuracy logging
        preds = torch.argmax(y_hat, dim=1)
        for class_name, class_idx in grouped_label_to_index.items():
            class_mask = y == class_idx
            if class_mask.sum() > 0:
                y_hat_class = y_hat[class_mask].to(self.device)
                y_class = y[class_mask].to(self.device)
                metric = getattr(self, f"test_acc_{class_name}")
                metric(y_hat_class, y_class)
                self.log(
                    f"test_acc_{class_name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"test_acc_{class_name}",
                )

        return loss

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.test_accuracy.compute())
        self.log("test_auroc_epoch", self.test_auroc.compute())
        # Handle or reset saved outputs as needed
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True)


# Main training loop
def train_model(
    downsample_factor,
    num_epochs=num_epochs,
    config=default_config,
    batch_size=batch_size,
    num_classes=num_classes,
):
    data_module = ImageDataModule(
        combined_metadata_csv_path="/media/hdd3/neo/new_plasma_cell_deepheme_training_metadata.csv",
        batch_size=batch_size,
        downsample_factor=downsample_factor,
    )

    model = Myresnext50(num_classes=num_classes)

    # Logger
    logger = TensorBoardLogger("lightning_logs", name=str(downsample_factor))

    # Trainer configuration for distributed training
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=num_gpus,
        accelerator="gpu",  # 'ddp' for DistributedDataParallel
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())


def model_create(path, num_classes=11):
    """
    Create a model instance from a given checkpoint.

    Parameters:
    - checkpoint_path (str): The file path to the PyTorch Lightning checkpoint.

    Returns:
    - model (Myresnext50): The loaded model ready for inference or further training.
    """
    # Instantiate the model with any required configuration
    # model = Myresnext50(
    #     num_classes=num_classes
    # )  # Adjust the number of classes if needed

    # # Load the model weights from a checkpoint
    model = Myresnext50.load_from_checkpoint(path)
    return model


def model_predict(model, pil_image):
    """
    Perform inference using the given model on the provided image.
    And return the softmax probabilities.
    """

    # Preprocess the image, by resizing and converting to tensor
    image = transforms.Resize((96, 96))(pil_image)
    image = transforms.ToTensor()(image)

    # add a batch dimension
    image = image.unsqueeze(0)

    # Perform inference
    model.eval()
    with torch.no_grad():

        # make sure both model and image are on the cuda device
        model.to("cuda")
        image = image.to("cuda")
        output = model(image)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # remove the batch dimension
    probabilities = probabilities.squeeze(0)

    # move the probabilities to the cpu
    probabilities = probabilities.cpu().numpy()

    # return the probabilities as a numpy array
    # assert the sum is within 1e-5 of 1
    assert (
        np.abs(probabilities.sum().item() - 1) < 1e-5
    ), "Probabilities do not sum to 1"

    return probabilities


############################################################################
########################## MODEL DEPLOYMENT ################################
############################################################################

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def predict_image(model, image_path):
    """
    Make prediction on a single image using CUDA

    Args:
        model: Loaded PyTorch model (must be on CUDA)
        image_path (str): Path to the image file

    Returns:
        tuple: (predicted_class_name, probabilities_dict)

    Raises:
        RuntimeError: If model is not on CUDA
    """
    # Ensure model is on CUDA
    if not next(model.parameters()).is_cuda:
        raise RuntimeError(
            "Model must be on CUDA. Please load model using load_model() function."
        )

    # Label mapping
    label_mapping = {
        0: "blasts and blast-equivalents",
        1: "promyelocytes",
        2: "myelocytes",
        3: "metamyelocytes",
        4: "neutrophils/bands",
        5: "monocytes",
        6: "eosinophils",
        7: "erythroid precursors",
        8: "lymphocytes",
        9: "plasma cells",
        10: "skippocytes",
    }

    # Prepare image transforms
    transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])

    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = transform(image)

    # Add batch dimension and move to CUDA immediately
    image = image.unsqueeze(0).cuda()

    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]

    # Move predictions to CPU for converting to Python types
    probabilities = probabilities.cpu()

    # Get predicted class and probabilities
    predicted_class = label_mapping[int(torch.argmax(probabilities))]
    probs_dict = {label_mapping[i]: float(prob) for i, prob in enumerate(probabilities)}

    return predicted_class, probs_dict
