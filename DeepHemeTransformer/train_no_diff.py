import os
import numpy as np
import pytorch_lightning as pl
from DeepHemeTransformer import DeepHemeModule
from cell_dataloader import CellFeaturesDataModule
from pytorch_lightning.loggers import TensorBoardLogger

for i in range(1):
    num_epoches = 10
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
    log_dir = f"../logs/train_nov25_no_diff/lr_{learning_rate}_no_diff_transformer"
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="",
    )

    # Define a PyTorch Lightning trainer with the custom logger
    trainer = pl.Trainer(
        max_epochs=num_epoches,
        log_every_n_steps=10,
        devices=1,
        accelerator="gpu",
        logger=logger,
    )

    # Create an instance of your LightningModule
    model = DeepHemeModule(
        learning_rate=learning_rate,
        max_epochs=num_epoches,
        weight_decay=1e-2,
        reg_lambda1=0,
        reg_lambda2=1,
    )

    # Train the model
    trainer.fit(model, datamodule)
