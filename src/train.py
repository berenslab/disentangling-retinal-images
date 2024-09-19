import argparse
import os

import torch
from omegaconf import OmegaConf

from src.dataset.eyepacs import EyePACS
from src.dataset.utils import compute_label_dims
from src.generative_model.stylegan import StyleGAN2Model
from src.generative_model.trainer import create_trainer
from src.utils.utils import get_labels, load_yaml_config, make_exp_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--train_config",
        type=str,
        help="name of yaml config file",
        default="configs/configs_train/test.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(config_filename=args.train_config)
    experiment_folder = make_exp_folder(config)
    config = OmegaConf.create(config)
    labels = get_labels(config)

    train_set = EyePACS(
        image_root_dir=config.data.image_root_dir,
        meta_factorized_path=config.data.meta_factorized_path,
        columns_mapping_path=config.data.columns_mapping_path,
        splits_dir=config.data.splits_dir,
        split="train",
        image_size=config.data.image_size,
        input_preprocessing=config.data.input_preprocessing,
        labels=labels,
        onehot_enc=False,
        subset=config.data.train_subset,
        filter_meta=config.data.filter_meta,
        ram=config.data.ram,
    )

    val_set = EyePACS(
        image_root_dir=config.data.image_root_dir,
        meta_factorized_path=config.data.meta_factorized_path,
        columns_mapping_path=config.data.columns_mapping_path,
        splits_dir=config.data.splits_dir,
        split="val",
        image_size=config.data.image_size,
        input_preprocessing=config.data.input_preprocessing,
        labels=labels,
        onehot_enc=False,
        subset=config.data.val_subset,
        filter_meta=config.data.filter_meta,
        ram=config.data.ram,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        config.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=True,
    )
    lambda_gp = (
        0.0002
        * (config.data.image_size**2)
        / (config.data.batch_size * len(config.gpus))
    )  # heuristic formula from original implementation

    cond_dims = compute_label_dims(train_set, config.data.conditional_labels)
    class_dims = compute_label_dims(train_set, config.data.classifier_labels)

    if len(cond_dims) > 0:
        _, counts = torch.unique(train_set._meta[:, 0], return_counts=True)
        cond_distribution = torch.distributions.categorical.Categorical(
            probs=counts / counts.sum()
        )
    else:
        cond_distribution = None

    trainer, checkpoint_callback = create_trainer(config, experiment_folder)
    model = StyleGAN2Model(
        config,
        experiment_folder,
        lambda_gp,
        cond_dims,
        class_dims,
        cond_distribution,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=config.resume,
    )

    with open(os.path.join(experiment_folder, "best_ckpt.txt"), "w") as text_file:
        text_file.write(checkpoint_callback.best_model_path)
