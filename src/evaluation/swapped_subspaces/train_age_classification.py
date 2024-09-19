import argparse
import os

import torch
from omegaconf import OmegaConf

from src.dataset.eyepacs import EyePACS
from src.evaluation.swapped_subspaces.classification_model import \
    ClassificationModel
from src.generative_model.trainer import create_trainer
from src.utils.utils import get_labels, load_yaml_config, make_exp_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--train_config",
        type=str,
        help="name of yaml config file",
        default="../configs/configs_swapped_subspaces/train_age_classification.yaml",
    )
    parser.add_argument(
        "-d",
        "--gan_experiment_folder",
        type=str,
        help="path to experiment folder of gan model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(config_filename=args.train_config)
    config["gan_experiment_folder"] = args.gan_experiment_folder
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

    trainer, checkpoint_callback = create_trainer(config, experiment_folder)
    model = ClassificationModel(
        resnet_backbone=config.resnet_backbone,
        weight_decay=config.weight_decay,
        num_classes=config.num_classes,
        gan_model_exp_folder=args.gan_experiment_folder,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=config.resume,
    )

    with open(os.path.join(experiment_folder, "best_ckpt.txt"), "w") as text_file:
        text_file.write(checkpoint_callback.best_model_path)
