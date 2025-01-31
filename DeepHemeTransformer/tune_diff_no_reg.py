import os
import numpy as np
import pytorch_lightning as pl
from DeepHemeTransformer import DeepHemeModule
from cell_dataloader import CellFeaturesDataModule
from pytorch_lightning.loggers import TensorBoardLogger

for i in range(10):
    num_epoches = 5
    learning_rate = 10 ** np.random.uniform(-10, 0)
    # Set up parameters
    metadata_file_path = (
        "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
    )
    batch_size = 32
    # Instantiate the DataModule
    datamodule = CellFeaturesDataModule(
        metadata_file=metadata_file_path, batch_size=batch_size
    )

    # # use a 1e-4 learning rate
    # learning_rate = 1e-4

    # Set up the logger with a subfolder named after the learning rate
    log_dir = f"logs/tune_nov3_v2/lr_{learning_rate}_no_reg"
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

    # Create an instance of your LightningModule # NOTE the max epochs is 20*num_epoches because we are trying to make sure the cosine scheduler does not actually reduce the learning rate we are using flat learning rate for tuning the model
    model = DeepHemeModule(
        learning_rate=learning_rate, max_epochs=20 * num_epoches, weight_decay=1e-2
    )

    # Train the model
    trainer.fit(model, datamodule)
