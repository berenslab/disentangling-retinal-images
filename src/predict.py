import argparse
import os

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from src.dataset.eyepacs import EyePACS
from src.dataset.utils import compute_label_dims
from src.generative_model.stylegan import StyleGAN2Model
from src.utils.utils import get_labels, load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--data_split",
        type=str,
        help="one of: train, val, test",
        default="train",
    )
    parser.add_argument(
        "-c",
        "--predict_config",
        type=str,
        help="name of yaml config file",
        default="configs/configs_predict/default.yaml",
    )
    parser.add_argument(
        "-d",
        "--experiment_folder",
        type=str,
        help="path to experiment folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.create(load_yaml_config(config_filename=args.predict_config))
    model_config = OmegaConf.create(
        load_yaml_config(
            os.path.join(args.experiment_folder, "config.yaml")
        )
    )
    print("Load data")
    labels = get_labels(model_config)
    dataset = EyePACS(
        image_root_dir=model_config.data.image_root_dir,
        meta_factorized_path=model_config.data.meta_factorized_path,
        columns_mapping_path=model_config.data.columns_mapping_path,
        splits_dir=model_config.data.splits_dir,
        split=args.data_split,
        image_size=model_config.data.image_size,
        input_preprocessing=model_config.data.input_preprocessing,
        labels=labels,
        onehot_enc=False,
        subset=config.data.subset,
        filter_meta=model_config.data.filter_meta,
        ram=config.data.ram,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        pin_memory=True,
        drop_last=False,
    )

    cond_dims = compute_label_dims(dataset, model_config.data.conditional_labels)
    c_dims = compute_label_dims(dataset, model_config.data.classifier_labels)

    model = StyleGAN2Model(
        config=model_config,
        experiment_folder=args.experiment_folder,
        cond_dims=cond_dims,
        class_dims=c_dims,
        lambda_gp=0,
    )

    with open(os.path.join(args.experiment_folder, "best_ckpt.txt"), "r") as file:
        checkpoint_file = file.read().rstrip()
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    trainer = Trainer(devices=config.gpus, accelerator="gpu")
    embeddings = trainer.predict(model, dataloaders=dataloader)

    embedding_folder = os.path.join(args.experiment_folder, "embeddings/")
    if not os.path.exists(embedding_folder):
        os.mkdir(embedding_folder)
    torch.save(
        embeddings,
        f=os.path.join(embedding_folder, f"{args.data_split}_embeddings.pt"),
    )
    torch.save(
        dataset._image_paths,
        f=os.path.join(embedding_folder, f"{args.data_split}_image_paths.pt"),
    )
