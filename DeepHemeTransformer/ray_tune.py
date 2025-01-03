import torch
import lightning as pl
from ray import tune
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.schedulers import ASHAScheduler
import os

from cell_dataloader import CellFeaturesDataModule
from DeepHemeTransformer import DeepHemeModule

def train_deep_heme(config):
    model = DeepHemeModule(
        learning_rate=config["learning_rate"],
        num_heads=config["num_heads"],
        reg_lambda=config["reg_lambda"],
        weight_decay=config["weight_decay"]
    )
    
    datamodule = CellFeaturesDataModule(
        metadata_file=config["metadata_file_path"],
        batch_size=config["batch_size"]
    )
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],  # Removed metrics parameter
        enable_progress_bar=False
    )
    
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=datamodule)

def main():
    config = {
        "num_heads": tune.choice([1, 2, 4, 8]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "reg_lambda": tune.uniform(0.01, 1.0),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "metadata_file_path": "/media/hdd3/neo/DeepHemeTransformerData/labelled_features_metadata.csv"
    }

    scheduler = ASHAScheduler(
        max_t=50,
        grace_period=10,
        metric="val_loss",
        mode="min"
    )

    analysis = tune.run(
        train_deep_heme,
        config=config,
        num_samples=50,
        scheduler=scheduler,
        resources_per_trial={"gpu": 1, "cpu": 2},
        # local_dir="./ray_results",
        name="deepheme_tune",
        stop={"training_iteration": 50}
    )

    print("Best config:", analysis.best_trial.config)
    print(f"Best validation loss: {analysis.best_trial.last_result['val_loss']:.4f}")
    
if __name__ == "__main__":
    main()