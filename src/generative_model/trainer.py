import os
from pathlib import Path
from random import randint

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

def create_trainer(config: dict, experiment_folder: str):
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed)

    # create logging folders
    tensorboard_folder = os.path.join(experiment_folder, "tensorboard")
    ckpt_folder = os.path.join(experiment_folder, "checkpoints")
    Path(tensorboard_folder).mkdir(parents=False, exist_ok=True)
    Path(ckpt_folder).mkdir(parents=False, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=tensorboard_folder,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_folder,
        save_top_k=-1,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        filename="_{epoch}",
    )
    print(f"batch_size = {config.data.batch_size}")

    trainer = Trainer(
        devices=config.gpus,
        accelerator="gpu",
        strategy=(
            DDPStrategy(find_unused_parameters=True) if len(config.gpus) > 1 else "auto"
        ),
        num_sanity_val_steps=config.sanity_steps,
        max_epochs=config.max_epoch,
        limit_val_batches=config.val_check_percent,
        callbacks=[checkpoint_callback],
        val_check_interval=float(min(config.val_check_interval, 1)),
        check_val_every_n_epoch=max(1, config.val_check_interval),
        logger=logger,
        benchmark=True,
        precision="16-mixed",
    )
    return trainer, checkpoint_callback
