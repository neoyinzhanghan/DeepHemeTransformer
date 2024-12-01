import models
from dataset import TensorStackDataModule
import schedulers
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torch import manual_seed as set_seed1
from random import seed as set_seed2
from numpy.random import seed as set_seed3

if __name__ == "__main__":

    # Set a seed to make the training deterministic
    seed = 4  # 1 3 4
    set_seed1(seed)
    set_seed2(seed)
    set_seed3(seed)
    deterministic = True

    lr = 1e-5
    scheduler = schedulers.LinearLR
    config = {"start_factor": 1, "end_factor": 0.5, "total_iters": 100}
    # scheduler = schedulers.MyCosineLR
    # config = {'max_lr':lr, 'base_lr':2e-4, 'T_max':50}
    # scheduler = schedulers.MyCyclicLR
    # config = {'max_lr':lr, 'base_lr':1e-6, 'step_size_up':20}
    # scheduler = schedulers.CyclicLR
    # config = {'base_lr':lr, 'max_lr':5.5e-4, 'mode':'triangular2', 'step_size_up':100, 'cycle_momentum':False}

    num_classes = 9
    init_mil_embed = 2048  # The initial embedding size of the MIL cells
    mil_head = 256  # The output size of the aggregation network
    attn_head_size = 128  # The output size of the attention network
    agg_method = "normal"
    args = (init_mil_embed, mil_head, attn_head_size, agg_method)
    model = models.MILSelfAttentionTwo(
        num_classes, args, lr=lr, scheduler=scheduler, config=config
    )
    # model = models.MILSelfAttentionOne(num_classes, args, lr=lr, scheduler=scheduler, config=config)

    name = f"mantle/{seed}"
    feature_stacks_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/feature_stacks"
    diff_data_path = (
        "/media/hdd3/neo/DiffTransformerV1DataMini/subsampled_split_diff_data.csv"
    )
    batch_size = 20
    num_epochs = 300
    num_gpus = (
        2  # The index of the gpu if in a list, otherwise the number to distribute among
    )
    num_workers = 16

    data_module = TensorStackDataModule(
        feature_stacks_dir=feature_stacks_dir,
        diff_data_path=diff_data_path,
        batch_size=batch_size,
        num_workers=8,
    )
    # Logger
    logger = TensorBoardLogger("lightning_logs/wbc_only", name=name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="best",  # "{epoch:02d}",
        monitor="val_loss",
        mode="max",
        save_last=True,
        save_top_k=1,  # -1 don't delete old checkpoints
    )

    # Trainer configuration for distributed training
    trainer = pl.Trainer(
        callbacks=checkpoint_callback,
        max_epochs=num_epochs,
        logger=logger,
        devices=num_gpus,
        accelerator="cuda",
    )
    trainer.fit(model, data_module.train_dataloader(), data_module.train_dataloader())
    trainer.test(
        ckpt_path=checkpoint_callback.best_model_path,
        dataloaders=data_module.train_dataloader(),
    )
