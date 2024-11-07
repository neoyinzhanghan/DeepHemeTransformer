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
from BMAassumptions import cellnames


############################################################################
####### DEFINE HYPERPARAMETERS AND DATA DIRECTORIES ########################
############################################################################

num_epochs = 500
default_config = {"lr": 3.56e-05}  # 1.462801279401232e-06}
base_data_dir = "/media/hdd3/neo/pooled_deepheme_data"
num_gpus = 2
num_workers = 64
downsample_factor = 1
batch_size = 256
img_size = 96
num_classes = 23

results_dirs_list = [
    "/media/hdd3/neo/AML_bma",
    "/media/hdd3/neo/b-all_bma",
    "/media/hdd3/neo/lpl_bma",
    "/media/hdd3/neo/myeloma_bma",
]

cell_types_list = ["M1", "M1", "L4", "L4"]

base_data_sample_probability = 0.5
sample_probabilities = [0.125, 0.125, 0.125, 0.125]

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
        base_data_dir,
        results_dirs_list,
        cell_types_list,
        batch_size,
        downsample_factor,
    ):
        super().__init__()
        self.data_dir = base_data_dir
        self.results_dirs_list = results_dirs_list
        self.cell_types_list = cell_types_list
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Additional normalization can be uncommented and adjusted if needed
                # transforms.Normalize(mean=(0.61070228, 0.54225375, 0.65411311), std=(0.1485182, 0.1786308, 0.12817113))
            ]
        )

    def setup(self, stage=None):
        # Load base train, validation, and test datasets
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"), transform=self.transform
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"), transform=self.transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "test"), transform=self.transform
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

        # Compute class weights for handling imbalance (for train dataset)
        class_counts_train = torch.tensor(
            [t[1] for t in train_dataset.samples]
        ).bincount()
        class_weights_train = 1.0 / class_counts_train.float()
        sample_weights_train = class_weights_train[
            [t[1] for t in train_dataset.samples]
        ]

        class_counts_val = torch.tensor([t[1] for t in val_dataset.samples]).bincount()
        class_weights_val = 1.0 / class_counts_val.float()
        sample_weights_val = class_weights_val[[t[1] for t in val_dataset.samples]]

        class_counts_test = torch.tensor(
            [t[1] for t in test_dataset.samples]
        ).bincount()
        class_weights_test = 1.0 / class_counts_test.float()
        sample_weights_test = class_weights_test[[t[1] for t in test_dataset.samples]]

        self.train_sampler = WeightedRandomSampler(
            weights=sample_weights_train,
            num_samples=len(sample_weights_train),
            replacement=True,
        )

        self.val_sampler = WeightedRandomSampler(
            weights=sample_weights_val,
            num_samples=len(sample_weights_val),
            replacement=True,
        )

        self.test_sampler = WeightedRandomSampler(
            weights=sample_weights_test,
            num_samples=len(sample_weights_test),
            replacement=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=24,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=24,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=24,
        )


# Model Module
class Myresnext50(pl.LightningModule):
    def __init__(self, num_classes=23, config=default_config):
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

        self.train_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.train_auroc = AUROC(num_classes=num_classes, task=task)
        self.val_auroc = AUROC(num_classes=num_classes, task=task)
        self.test_accuracy = Accuracy(num_classes=num_classes, task=task)
        self.test_auroc = AUROC(num_classes=num_classes, task=task)

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
    base_data_dir=base_data_dir,
    results_dirs_list=results_dirs_list,
    cell_types_list=cell_types_list,
    batch_size=batch_size,
    num_classes=num_classes,
):
    data_module = ImageDataModule(
        base_data_dir=base_data_dir,
        results_dirs_list=results_dirs_list,
        cell_types_list=cell_types_list,
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


def model_create(path, num_classes=23):
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
    # pil_image = transforms.Resize((96, 96))(pil_image)
    image = transforms.ToTensor()(pil_image)

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


def predict_batch(pil_images, model):
    # Define the transformations
    image_transforms = transforms.Compose(
        [
            # transforms.Resize(96),
            transforms.ToTensor(),
        ]
    )

    # Apply transformations to each image and create a batch
    batch = torch.stack([image_transforms(image).float() for image in pil_images])

    # Move the batch to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)

    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    # Process each output as in the original code snippet
    predictions = []
    for output in outputs:
        # apply softmax to the output
        output = F.softmax(output, dim=0)
        output = output.detach().cpu().numpy()
        predictions.append(tuple(output))

    # Return a list of predictions in the same order as the input images
    return predictions


def model_predict_batch(model, pil_images):
    """
    Perform inference using the given model on a batch of provided images.
    Returns the batched stacked softmax probabilities.
    """
    # Preprocess the images: resize and convert to tensor, then stack into a batch
    # pil_images = [transforms.Resize((96, 96))(img) for img in pil_images]
    images = [transforms.ToTensor()(img) for img in pil_images]
    images = torch.stack(images)  # Creates a batch of images

    # Perform inference
    model.eval()
    with torch.no_grad():
        # Make sure both model and images are on the CUDA device
        model.to("cuda")
        images = images.to("cuda")
        output = model(images)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # Move the probabilities to the CPU and convert to numpy
    probabilities = probabilities.cpu().numpy()

    # Assert that the sum of probabilities for each image is approximately 1
    assert np.allclose(
        probabilities.sum(axis=1), 1, atol=1e-5
    ), "Probabilities do not sum to 1"

    list_of_probabilities = predict_batch(pil_images, model)

    print("Example Output")
    print(probabilities[0])
    print(list_of_probabilities[0])

    # for prob in probabilities:
    #     for i in range(len(prob)):
    #         print(
    #             f"{cellnames[i]}: {prob[i]}"
    #         )  # TODO remove this line, this is for debugging only

    return probabilities


def extract_features_batch(model, pil_images):
    """
    Extract features using the given model on a batch of provided images.
    Returns the batched stacked features.
    """
    # Preprocess the images: resize and convert to tensor, then stack into a batch
    # pil_images = [transforms.Resize((96, 96))(img) for img in pil_images]
    images = [transforms.ToTensor()(img) for img in pil_images]
    images = torch.stack(images)  # Creates a batch of images

    # Perform feature extraction
    model.eval()
    with torch.no_grad():
        # Make sure both model and images are on the CUDA device
        model.to("cuda")
        images = images.to("cuda")
        features = model.extract_features(images)

    # Move the features to the CPU and convert to numpy
    features = features.cpu().numpy()

    return features


if __name__ == "__main__":
    # Run training for each downsampling factor

    # Train the model
    # train_model(downsample_factor=downsample_factor)

    from BMAassumptions import HemeLabel_ckpt_path
    from PIL import Image

    model = model_create(HemeLabel_ckpt_path)

    test_image = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir/BMA-diff_2024-10-05 14:55:49/cells/ER1/ER1-L2-U1-M1_1223-7.jpg"

    # Load the test image
    image = Image.open(test_image)

    # generate an image that is 96x96 that is just black
    image = Image.new("RGB", (96, 96), color="black")

    # Perform inference
    probabilities = model_predict(model, image)

    print("Example Output")
    print(probabilities)
