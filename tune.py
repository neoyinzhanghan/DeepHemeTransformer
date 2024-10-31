import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from cell_dataloader import CellFeaturesDataModule  # assuming this module is correctly implemented
from loss_fn import RegularizedDifferentialLoss
from DeepHemeTransformer import DeepHemeModule  # assuming this is your defined model
import pandas as pd


def train_deep_heme_module(config, metadata_file_path, num_epochs=50):
    # Set up data module
    datamodule = CellFeaturesDataModule(
        metadata_file=metadata_file_path,
        batch_size=32
    )

    # Set up model with config parameters
    model = DeepHemeModule(
        learning_rate=config["learning_rate"],
        max_epochs=num_epochs,
        weight_decay=1e-2,  # Static value, modify if you want to tune it as well
        num_heads=config["num_heads"],
        reg_lambda=config["reg_lambda"]
    )

    # Set up trainer with Ray Tune callback
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=10,
        devices=1,
        accelerator="gpu",
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")],
    )

    # Train the model
    trainer.fit(model, datamodule)


# Define the search space for random search
search_space = {
    "num_heads": tune.choice([1, 2, 4, 8]),
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "reg_lambda": tune.uniform(0.01, 1.0),
}

# Metadata file path for data module
metadata_file_path = "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"

# Run the tuning
tuner = tune.Tuner(
    tune.with_parameters(train_deep_heme_module, metadata_file_path=metadata_file_path),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        num_samples=50,  # Set the number of random search samples
    ),
    run_config=tune.RunConfig(name="deep_heme_random_search"),
)

# Run the tuning and save results to CSV
result = tuner.fit()
results_df = result.get_dataframe()
results_df.to_csv("tune_results.csv", index=False)
