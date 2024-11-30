import models
import schedulers
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torch import manual_seed as set_seed1
from random import seed as set_seed2
from numpy.random import seed as set_seed3
from torch.backends.cudnn import deterministic

if __name__ == "__main__":

    # Set a seed to make the training deterministic
    seed = 1
    set_seed1(seed)
    set_seed2(seed)
    set_seed3(seed)
    deterministic = True
    
    project = 'mortality'
    lr = 1e-4
    scheduler = schedulers.LinearLR
    config = {'start_factor':1, 'end_factor':.5, 'total_iters':100}
    #scheduler = schedulers.MyCosineLR
    #config = {'max_lr':lr, 'base_lr':2e-4, 'T_max':50}
    #scheduler = schedulers.MyCyclicLR
    #config = {'max_lr':lr, 'base_lr':1e-6, 'step_size_up':20}
    #scheduler = schedulers.CyclicLR
    #config = {'base_lr':lr, 'max_lr':5.5e-4, 'mode':'triangular2', 'step_size_up':100, 'cycle_momentum':False}
    num_classes, args, model = models.MODEL_DICT[project]
    model = model(num_classes, args, lr=lr, scheduler=scheduler, config=config)
    #args = (init_mil_embed, mil_head, attn_head_size, agg_method)
    #model = models.MILSelfAttentionTwo(num_classes, args, lr=lr, scheduler=scheduler, config=config)

    
    experiment = '1'
    patient_data_dir=f'/media/ssd2/clinical_text_data/Mortality/experiments/{experiment}'
    image_data_dir='/media/ssd2/clinical_text_data/Mortality/Patients'
    batch_size = 32
    num_epochs = 200
    num_gpus = [1] # The index of the gpu if in a list, otherwise the number to distribute among
    num_workers = 10
    
    data_module = utils.TensorDataModule(
        patient_data_dir=patient_data_dir, 
        image_data_dir=image_data_dir, 
        batch_size=batch_size, 
        num_workers=num_workers
    )

    # Logger
    logger = TensorBoardLogger("lightning_logs", name=f'{project}/{experiment}/{seed}')
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='best', #"{epoch:02d}",
        monitor='val_auroc',
        mode='max',
        save_last=True,
        save_top_k=1,  #-1 don't delete old checkpoints
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        mode='min'
    )

    # Trainer configuration for distributed training
    trainer = pl.Trainer(
        callbacks=[model_checkpoint_callback, early_stopping_callback],
        max_epochs=num_epochs,
        logger=logger,
        devices=num_gpus,
        accelerator='ddp' if type(num_gpus) == int and num_gpus > 1 else 'gpu',  # 'ddp' for DistributedDataParallel
    )
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    trainer.test(ckpt_path=model_checkpoint_callback.best_model_path, 
                 dataloaders=data_module.test_dataloader())
