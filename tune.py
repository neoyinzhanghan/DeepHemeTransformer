import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pandas as pd
from pathlib import Path
import os

# Assuming these are implemented in the same directory
from cell_dataloader import CellFeaturesDataModule
from loss_fn import RegularizedDifferentialLoss
from DeepHemeTransformer import DeepHemeModule

def train_deep_heme_module(config, metadata_file_path, num_epochs=50):
    """
    Training function for DeepHeme model with multi-GPU support.
    
    Args:
        config (dict): Hyperparameter configuration from Ray Tune
        metadata_file_path (str): Path to the metadata file
        num_epochs (int): Number of training epochs
    """
    # Set seeds for reproducibility
    pl.seed_everything(42)
    
    # Set up data module with error handling
    try:
        datamodule = CellFeaturesDataModule(
            metadata_file=metadata_file_path,
            batch_size=config.get("batch_size", 32),
            num_workers=4  # Added explicit num_workers for better GPU utilization
        )
    except Exception as e:
        print(f"Error initializing DataModule: {str(e)}")
        raise
    
    # Initialize model with configuration
    model = DeepHemeModule(
        learning_rate=config["learning_rate"],
        max_epochs=num_epochs,
        weight_decay=config.get("weight_decay", 1e-2),
        num_heads=config["num_heads"],
        reg_lambda=config["reg_lambda"]
    )
    
    # Configure training callbacks
    callbacks = [
        TuneReportCallback(
            {
                "loss": "val_loss",
                "accuracy": "val_accuracy"
            },
            on="validation_end"
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min"
        )
    ]
    
    # Get GPU device index from Ray's resource specification
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"Training on GPU {gpu_id}")
    
    # Initialize trainer with GPU configuration
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=10,
        accelerator="gpu",
        devices=[int(gpu_id)],  # Use the GPU assigned by Ray
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=False,  # Disable progress bar for cleaner logging
        precision="32-true"  # Use full precision for debugging
    )
    
    # Train the model with error handling
    try:
        trainer.fit(model, datamodule)
    except Exception as e:
        print(f"Training error on GPU {gpu_id}: {str(e)}")
        raise

def main():
    # Print available GPU information
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Define the search space for random search
    search_space = {
        "num_heads": tune.choice([1, 2, 4, 8]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "reg_lambda": tune.uniform(0.01, 1.0),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-5, 1e-2)
    }
    
    # Configure metadata file path
    metadata_file_path = Path("/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv")
    
    # Verify metadata file exists
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")
    
    # Configure resources per trial
    resources_per_trial = {"gpu": 1}  # Request 1 GPU per trial
    
    # Configure Ray Tune
    tune_config = tune.TuneConfig(
        metric="loss",
        mode="min",
        num_samples=50,
        max_concurrent_trials=2,  # Run 2 trials in parallel (one per GPU)
    )
    
    # Initialize and run tuner with resource configuration
    tuner = tune.Tuner(
        tune.with_parameters(
            train_deep_heme_module,
            metadata_file_path=str(metadata_file_path)
        ),
        param_space=search_space,
        tune_config=tune_config,
        run_config=tune.RunConfig(
            storage_path="./ray_results",
            name="deep_heme_multi_gpu",
            resources_per_trial=resources_per_trial
        )
    )
    
    # Run hyperparameter search
    try:
        result = tuner.fit()
        
        # Save results to CSV
        results_df = result.get_dataframe()
        results_df.to_csv("ray_results/tune_results.csv", index=False)
        
        # Print best configuration
        best_trial = result.get_best_trial("loss", "min", "last")
        print("\nBest trial config:", best_trial.config)
        print("Best trial final validation loss:", best_trial.last_result["loss"])
        
    except Exception as e:
        print(f"Tuning error: {str(e)}")
        raise

if __name__ == "__main__":
    main()