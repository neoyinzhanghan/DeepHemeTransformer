import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
import pandas as pd
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming these are implemented in the same directory
from cell_dataloader import CellFeaturesDataModule
from loss_fn import RegularizedDifferentialLoss
from DeepHemeTransformer import DeepHemeModule


class CustomTuneReportCallback(TuneReportCallback):
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def train_deep_heme_module(config, metadata_file_path, num_epochs=50):
    """
    Training function for DeepHeme model with multi-GPU support.
    """
    pl.seed_everything(42)

    # GPU setup with error handling
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Get available GPU index from Ray
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        gpu_id = int(gpu_ids.split(",")[0])  # Use first GPU if multiple are specified

        torch.cuda.set_device(gpu_id)  # Set the GPU device explicitly
        logger.info(f"Training on GPU {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")

    except Exception as e:
        logger.error(f"GPU setup error: {str(e)}")
        raise

    try:
        datamodule = CellFeaturesDataModule(
            metadata_file=metadata_file_path,
            batch_size=config.get("batch_size", 32),
            num_workers=2,
        )
    except Exception as e:
        logger.error(f"DataModule initialization error: {str(e)}")
        raise

    model_config = {
        "learning_rate": config["learning_rate"],
        "max_epochs": num_epochs,
        "weight_decay": config.get("weight_decay", 1e-2),
        "num_heads": config["num_heads"],
        "reg_lambda": config["reg_lambda"],
    }

    try:
        model = DeepHemeModule(**model_config)
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}")
        raise

    callbacks = [
        CustomTuneReportCallback(
            {"loss": "val_loss", "accuracy": "val_accuracy"}, on="validation_end"
        ),
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss", patience=5, mode="min"
        ),
    ]

    trainer_config = {
        "max_epochs": num_epochs,
        "accelerator": "gpu",
        "devices": [gpu_id],
        "callbacks": callbacks,
        "enable_progress_bar": False,
    }

    try:
        trainer = pl.Trainer(**trainer_config)
        trainer.fit(model, datamodule)
    except Exception as e:
        logger.error(f"Training error on GPU {gpu_id}: {str(e)}")
        raise


def main():
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(
            f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
        )

    search_space = {
        "num_heads": tune.choice([1, 2, 4, 8]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "reg_lambda": tune.uniform(0.01, 1.0),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
    }

    metadata_file_path = Path(
        "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
    )
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")

    # Create absolute path for Ray results
    current_dir = os.path.abspath(os.path.dirname(__file__))
    ray_results_path = os.path.abspath(os.path.join(current_dir, "ray_results"))
    os.makedirs(ray_results_path, exist_ok=True)

    run_config = RunConfig(
        storage_path=ray_results_path,
        name="deep_heme_multi_gpu",
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        ),
    )

    scheduler = ASHAScheduler(max_t=50, grace_period=10, reduction_factor=2)

    tune_config = tune.TuneConfig(
        metric="loss",
        mode="min",
        num_samples=50,
        scheduler=scheduler,
        max_concurrent_trials=2,
    )

    try:
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    train_deep_heme_module, metadata_file_path=str(metadata_file_path)
                ),
                resources={"gpu": 1},
            ),
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )

        result = tuner.fit()

        # Save results
        results_df = result.get_dataframe()
        results_df.to_csv(
            os.path.join(ray_results_path, "tune_results.csv"), index=False
        )

        best_trial = result.get_best_trial("loss", "min", "last")
        logger.info("\nBest trial config:")
        logger.info(best_trial.config)
        logger.info(
            f"Best trial final validation loss: {best_trial.last_result['loss']:.4f}"
        )

        with open(os.path.join(ray_results_path, "best_config.txt"), "w") as f:
            for key, value in best_trial.config.items():
                f.write(f"{key}: {value}\n")

    except Exception as e:
        logger.error(f"Tuning error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
