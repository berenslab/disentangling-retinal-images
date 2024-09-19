import argparse
import os
import random

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from src.dataset.eyepacs import EyePACS
from src.evaluation.swapped_subspaces.classification_model import \
    ClassificationModel
from src.utils import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--test_config",
        type=str,
        help="name of yaml config file",
        default="../configs/configs_swapped_subspaces/test_age_classification.yaml",
    )
    parser.add_argument(
        "-d",
        "--classifier_exp_folder",
        type=str,
        help="path to experiment folder of classifier model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.create(utils.load_yaml_config(config_filename=args.test_config))
    model_config = OmegaConf.create(
        utils.load_yaml_config(
            os.path.join(args.classifier_exp_folder, "config.yaml")
        )
    )
    labels = utils.get_labels(config)

    classifier_exp_folder = args.classifier_exp_folder

    model = ClassificationModel(
        resnet_backbone=model_config.resnet_backbone,
        weight_decay=model_config.weight_decay,
        num_classes=model_config.num_classes,
        gan_model_exp_folder=model_config.gan_experiment_folder,
    )

    with open(os.path.join(classifier_exp_folder, "best_ckpt.txt"), "r") as file:
        checkpoint_file = file.read().rstrip()
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Testing on best model.
    test_set = EyePACS(
        image_root_dir=config.data.image_root_dir,
        meta_factorized_path=config.data.meta_factorized_path,
        columns_mapping_path=config.data.columns_mapping_path,
        splits_dir=config.data.splits_dir,
        split="test",
        image_size=config.data.image_size,
        input_preprocessing=config.data.input_preprocessing,
        labels=labels,
        onehot_enc=False,
        subset=config.data.test_subset,
        filter_meta=config.data.filter_meta,
        ram=config.data.ram,
    )
    # Shuffle test set deterministically.
    shuffle_indices = list(range(len(test_set)))
    random.Random(4).shuffle(shuffle_indices)
    test_set_shuffled = torch.utils.data.Subset(test_set, indices=shuffle_indices)

    test_dataloader = torch.utils.data.DataLoader(
        test_set_shuffled,
        config.data.batch_size_test,
        shuffle=False,
        pin_memory=True,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=True,
    )

    logger = CSVLogger(save_dir=os.path.join(classifier_exp_folder, config.csv_name))
    trainer = Trainer(
        devices=config.gpus,
        accelerator="gpu",
        logger=logger,
        strategy=(
            DDPStrategy(find_unused_parameters=True) if len(config.gpus) > 1 else "auto"
        ),
    )
    trainer.test(model, dataloaders=test_dataloader)
