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
from DeepHemeTransformer import DeepHemeModule
from cell_dataloader import CellFeaturesDataModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_deep_heme_module(config, metadata_file_path, num_epochs=50):
    """
    Training function for DeepHeme model with multi-GPU support.
    """
    pl.seed_everything(42)
    
    # GPU setup with error handling
    try:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        logger.info(f"Training on GPU {gpu_id} - {torch.cuda.get_device_name(int(gpu_id))}")
    except Exception as e:
        logger.error(f"GPU setup error: {str(e)}")
        raise

    # DataModule setup with validation
    try:
        datamodule = CellFeaturesDataModule(
            metadata_file=metadata_file_path,
            batch_size=config.get("batch_size", 32),
            num_workers=os.cpu_count() // 2  # Adjust workers based on CPU cores
        )
        datamodule.prepare_data()  # Validate data loading
    except Exception as e:
        logger.error(f"DataModule initialization error: {str(e)}")
        raise

    # Model initialization with config validation
    model_config = {
        "learning_rate": config["learning_rate"],
        "max_epochs": num_epochs,
        "weight_decay": config.get("weight_decay", 1e-2),
        "num_heads": config["num_heads"],
        "reg_lambda": config["reg_lambda"]
    }
    
    try:
        model = DeepHemeModule(**model_config)
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}")
        raise

    # Enhanced callbacks
    callbacks = [
        TuneReportCallback(
            metrics={
                "loss": "val_loss",
                "accuracy": "val_accuracy"
            },
            on="validation_end"
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            min_delta=1e-4
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best_model"
        )
    ]

    # Trainer configuration with memory optimization
    trainer_config = {
        "max_epochs": num_epochs,
        "log_every_n_steps": 10,
        "accelerator": "gpu",
        "devices": [int(gpu_id)],
        "callbacks": callbacks,
        "deterministic": True,
        "enable_progress_bar": False,
        "precision": "32-true",
        "gradient_clip_val": 1.0,  # Prevent exploding gradients
        "accumulate_grad_batches": config.get("accumulate_grad_batches", 1),
        "strategy": "auto"  # Let PyTorch Lightning choose the best strategy
    }

    try:
        trainer = pl.Trainer(**trainer_config)
        trainer.fit(model, datamodule)
    except Exception as e:
        logger.error(f"Training error on GPU {gpu_id}: {str(e)}")
        raise

def main():
    # System information logging
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    # Enhanced search space
    search_space = {
        "num_heads": tune.choice([1, 2, 4, 8]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "reg_lambda": tune.uniform(0.01, 1.0),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "accumulate_grad_batches": tune.choice([1, 2, 4])  # For larger effective batch sizes
    }

    metadata_file_path = Path("/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv")
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")

    # Updated Ray configuration
    run_config = RunConfig(
        storage_path="./ray_results",  # Updated from local_dir
        name="deep_heme_multi_gpu",
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min"
        )
    )

    # ASHA scheduler for efficient hyperparameter search
    scheduler = ASHAScheduler(
        max_t=50,  # max epochs
        grace_period=10,  # min epochs before pruning
        reduction_factor=2,
        brackets=3
    )

    tune_config = tune.TuneConfig(
        metric="loss",
        mode="min",
        num_samples=50,
        scheduler=scheduler,
        max_concurrent_trials=2  # One trial per GPU
    )

    try:
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    train_deep_heme_module,
                    metadata_file_path=str(metadata_file_path)
                ),
                resources={"gpu": 1}  # Request 1 GPU per trial
            ),
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config
        )

        result = tuner.fit()
        
        # Save and analyze results
        results_df = result.get_dataframe()
        results_df.to_csv("ray_results/tune_results.csv", index=False)
        
        best_trial = result.get_best_trial("loss", "min", "last")
        logger.info("\nBest trial config:")
        logger.info(best_trial.config)
        logger.info(f"Best trial final validation loss: {best_trial.last_result['loss']:.4f}")
        
        # Save best configuration
        with open("ray_results/best_config.txt", "w") as f:
            for key, value in best_trial.config.items():
                f.write(f"{key}: {value}\n")
            
    except Exception as e:
        logger.error(f"Tuning error: {str(e)}")
        raise

if __name__ == "__main__":
    main()