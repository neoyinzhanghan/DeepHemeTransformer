import models
import pytorch_lightning as pl
from my_schedulers import MyCosineWarmupLR
from pytorch_lightning.loggers import TensorBoardLogger
from CDdataset import TensorStackDataModuleV5
from torch import manual_seed as set_seed1
from random import seed as set_seed2
from numpy.random import seed as set_seed3
from torch.backends.cudnn import deterministic
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR, CyclicLR


if __name__ == "__main__":

    # Set a seed to make the training deterministic
    seed = 1
    set_seed1(seed)
    set_seed2(seed)
    set_seed3(seed)
    deterministic = True

    project = "myeloma"
    base_lr = 1e-7
    max_lr = 1e-4
    # scheduler = schedulers.MyCosineLR
    # config = {'max_lr':lr, 'base_lr':2e-4, 'T_max':50}
    # scheduler = schedulers.MyCyclicLR
    # config = {'max_lr':lr, 'base_lr':1e-6, 'step_size_up':20}
    # scheduler = schedulers.CyclicLR
    # config = {'base_lr':lr, 'max_lr':5.5e-4, 'mode':'triangular2', 'step_size_up':100, 'cycle_momentum':False}

    # args = (init_mil_embed, mil_head, attn_head_size, agg_method)
    # model = models.MILSelfAttentionTwo(num_classes, args, lr=lr, scheduler=scheduler, config=config)

    experiment = "1"
    # Example paths
    feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"  # Replace with your directory
    logit_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"  # Replace with your directory
    diff_data_path = "/media/hdd3/neo/DiffTransformerV1DataMini/split_diff_data.csv"  # Replace with your CSV file
    dx_data_path = "/media/hdd3/neo/dx_data_test.csv"  # Replace with your CSV file
    batch_size = 16
    num_epochs = 1000
    num_gpus = [
        1
    ]  # The index of the gpu if in a list, otherwise the number to distribute among
    num_workers = 10

    data_module = TensorStackDataModuleV5(
        feature_stacks_dir=feature_stacks_dir,
        logit_stacks_dir=logit_stacks_dir,
        diff_data_path=diff_data_path,
        dx_data_path=dx_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # define a cosine annealing decay scheduler
    scheduler = MyCosineWarmupLR

    config = {
        "max_lr": max_lr,
        "base_lr": base_lr,
        "warmup_epochs": 10,
        "T_max": num_epochs,
    }

    data_module.setup()

    num_classes, args, model = models.MODEL_DICT[project]
    model = model(num_classes, args, lr=max_lr, scheduler=scheduler, config=config)

    # Logger
    logger = TensorBoardLogger("lightning_logs", name=f"{project}/{experiment}/{seed}")
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="best",  # "{epoch:02d}",
        monitor="val_auroc",
        mode="max",
        save_last=True,
        save_top_k=1,  # -1 don't delete old checkpoints
    )
    # early_stopping_callback = pl.callbacks.EarlyStopping(
    #     monitor="val_loss", patience=25, mode="min"
    # )

    # Trainer configuration for distributed training
    trainer = pl.Trainer(
        callbacks=[model_checkpoint_callback],  # early_stopping_callback],
        max_epochs=num_epochs,
        logger=logger,
        devices=num_gpus,
        accelerator=(
            "ddp" if type(num_gpus) == int and num_gpus > 1 else "gpu"
        ),  # 'ddp' for DistributedDataParallel
    )
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    trainer.test(
        ckpt_path=model_checkpoint_callback.best_model_path,
        dataloaders=data_module.test_dataloader(),
    )
