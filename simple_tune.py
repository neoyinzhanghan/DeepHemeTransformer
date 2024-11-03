import os
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from datetime import datetime
import torch

from cell_dataloader import CellFeaturesDataModule
from DeepHemeTransformer import DeepHemeModule


def get_random_config():
    """Generate random hyperparameters within defined ranges."""
    return {
        "num_heads": random.choice([1, 2, 4, 8]),
        "learning_rate": float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3)))),
        "reg_lambda": 1,  # random.uniform(0.01, 1.0),
        "batch_size": random.choice([16, 32, 64, 128]),
        "weight_decay": float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2)))),
    }


def train_model(config, metadata_file_path):
    """
    Train a single model with given hyperparameters.

    Args:
        config: Dictionary of hyperparameters
        metadata_file_path: Path to the metadata file

    Returns:
        dict: Training results including validation loss and other metrics
    """
    # Set seeds for reproducibility
    pl.seed_everything(42)

    # Initialize model
    model = DeepHemeModule(
        learning_rate=config["learning_rate"],
        num_heads=config["num_heads"],
        reg_lambda=config["reg_lambda"],
        weight_decay=config["weight_decay"],
    )

    # Initialize data module
    datamodule = CellFeaturesDataModule(
        metadata_file=metadata_file_path, batch_size=config["batch_size"]
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=1,  # should be 50
        accelerator="gpu",
        devices=2,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1, filename="best_model"
            ),
        ],
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    # Get best validation loss
    best_val_loss = trainer.callback_metrics.get(
        "val_loss", torch.tensor(float("inf"))
    ).item()

    return {
        "val_loss": best_val_loss,
        "epochs_completed": trainer.current_epoch + 1,
        "early_stopped": trainer.current_epoch + 1 < trainer.max_epochs,
    }


def main():
    """Run random hyperparameter search and save results."""
    # Configuration
    num_trials = 5
    metadata_file_path = (
        "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
    )

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("random_search_results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize results list
    results = []

    # Run random search
    best_val_loss = float("inf")
    best_config = None

    print(f"Starting random search with {num_trials} trials...")

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")

        # Generate random configuration
        config = get_random_config()
        print("Current configuration:", config)

        # Train model and get results
        try:
            trial_results = train_model(config, metadata_file_path)

            # Combine config and results
            trial_data = {"trial": trial + 1, **config, **trial_results}

            # Update best results
            if trial_results["val_loss"] < best_val_loss:
                best_val_loss = trial_results["val_loss"]
                best_config = config
                print(f"New best validation loss: {best_val_loss:.4f}")

            # Add to results list
            results.append(trial_data)

            # Save intermediate results to CSV
            pd.DataFrame(results).to_csv(
                os.path.join(results_dir, "random_search_results.csv"), index=False
            )

        except Exception as e:
            print(f"Error in trial {trial + 1}: {str(e)}")
            continue

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(results_dir, "random_search_results.csv"), index=False
    )

    # Save best configuration
    pd.DataFrame([best_config]).to_csv(
        os.path.join(results_dir, "best_config.csv"), index=False
    )

    print("\nRandom search completed!")
    print(f"Results saved in: {results_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best configuration:", best_config)


if __name__ == "__main__":
    main()
