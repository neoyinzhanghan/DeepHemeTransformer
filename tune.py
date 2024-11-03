import math
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
import pandas as pd
from pathlib import Path
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Assuming these are implemented in the same directory
from cell_dataloader import CellFeaturesDataModule
from loss_fn import RegularizedDifferentialLoss
from DeepHemeTransformer import DeepHemeModule

def train_deep_heme_module(config: Dict[str, Any], num_epochs: int = 50) -> None:
    """
    Training function for DeepHeme model following Ray Lightning integration best practices.
    Args:
        config: Hyperparameter configuration from Ray Tune
        num_epochs: Maximum number of training epochs
    """
    # Create data module
    datamodule = CellFeaturesDataModule(
        metadata_file=config["metadata_file_path"],
        batch_size=config["batch_size"],
        num_workers=2,
    )

    # Initialize model with config parameters
    model = DeepHemeModule(
        learning_rate=config["learning_rate"],
        max_epochs=num_epochs,
        weight_decay=config["weight_decay"],
        num_heads=config["num_heads"],
        reg_lambda=config["reg_lambda"],
    )

    # Callbacks following Ray's documentation
    callbacks = [
        TuneReportCallback(
            {
                "loss": "val_loss",
                "mean_accuracy": "val_accuracy"
            },
            on="validation_end"
        ),
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min"
        ),
    ]

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=False,
        deterministic=True,  # For reproducibility
    )

    trainer.fit(model, datamodule=datamodule)

def main():
    # Log available GPUs
    if torch.cuda.is_available():
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Define search space
    config = {
        "num_heads": tune.choice([1, 2, 4, 8]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "reg_lambda": tune.uniform(0.01, 1.0),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "metadata_file_path": os.path.abspath(
            "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
        ),
    }

    # Verify metadata file exists
    if not os.path.exists(config["metadata_file_path"]):
        raise FileNotFoundError(f"Metadata file not found at {config['metadata_file_path']}")

    # Configure scheduler
    scheduler = ASHAScheduler(
        max_t=50,
        grace_period=10,
        reduction_factor=2
    )

    # Ray Tune reporting metrics
    reporter = tune.CLIReporter(
        parameter_columns=["num_heads", "learning_rate", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )

    # Training with Ray Tune
    try:
        trainable = tune.with_parameters(
            train_deep_heme_module,
            num_epochs=50
        )

        analysis = tune.run(
            trainable,
            config=config,
            metric="loss",
            mode="min",
            scheduler=scheduler,
            progress_reporter=reporter,
            num_samples=50,
            resources_per_trial={
                "cpu": 2,
                "gpu": 1 if torch.cuda.is_available() else 0
            },
            name="deep_heme_tune",
            local_dir="./ray_results",
            fail_fast=True,  # For debugging
        )

        # Get and log best trial results
        best_trial = analysis.best_trial
        logger.info("Best trial config:")
        logger.info(best_trial.config)
        logger.info(f"Best trial final validation loss: {best_trial.last_result['loss']:.4f}")
        logger.info(f"Best trial final validation accuracy: {best_trial.last_result['mean_accuracy']:.4f}")

        # Save results
        best_config_path = os.path.join("./ray_results", "best_config.txt")
        with open(best_config_path, "w") as f:
            for key, value in best_trial.config.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nBest validation loss: {best_trial.last_result['loss']:.4f}\n")
            f.write(f"Best validation accuracy: {best_trial.last_result['mean_accuracy']:.4f}\n")

        # Save full results DataFrame
        analysis_df = analysis.results_df
        analysis_df.to_csv(os.path.join("./ray_results", "all_trials.csv"))

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()