import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
import torchvision
from cleanfid import fid
from omegaconf import OmegaConf

from src.dataset.utils import compute_label_dims
from src.dataset.eyepacs import EyePACS
from src.generative_model.stylegan import StyleGAN2Model
from src.utils.utils import get_labels, load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="name of yaml config file",
        default="../configs/configs_image_quality/default.yaml",
    )
    parser.add_argument(
        "-d",
        "--experiment_folder",
        type=str,
        help="path to model experiment folder",
    )
    return parser.parse_args()


def export_real_images(output_dir_fid, dataloader):
    for iter_idx, batch in enumerate(dataloader):
        batch_image_real = batch["image"]
        for batch_idx in range(batch_image_real.shape[0]):
            torchvision.utils.save_image(
                batch_image_real[batch_idx],
                output_dir_fid / f"{iter_idx}_{batch_idx}.jpg",
                value_range=(-1, 1),
                normalize=True,
            )


def export_fake_images(
    model,
    output_dir_fid,
    num_images,
    batch_size,
    device,
    latent_dim=128,
):
    with torch.no_grad():
        grid_z = torch.randn(num_images, latent_dim)
        for iter_idx, latent in enumerate(grid_z.split(batch_size)):
            latent = latent.to(device)
            fake = model.G(latent, None, noise_mode="const").detach().cpu()
            for batch_idx in range(fake.shape[0]):
                torchvision.utils.save_image(
                    fake[batch_idx],
                    output_dir_fid / f"{iter_idx}_{batch_idx}.jpg",
                    value_range=(-1, 1),
                    normalize=True,
                )


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.create(load_yaml_config(config_filename=args.config_file))
    model_config = OmegaConf.create(
        load_yaml_config(
            os.path.join(args.experiment_folder, "config.yaml")
        )
    )

    labels = get_labels(model_config)

    print("Load data")
    dataset = EyePACS(
        image_root_dir=model_config.data.image_root_dir,
        meta_factorized_path=model_config.data.meta_factorized_path,
        columns_mapping_path=model_config.data.columns_mapping_path,
        splits_dir=model_config.data.splits_dir,
        split="test",
        image_size=model_config.data.image_size,
        input_preprocessing=model_config.data.input_preprocessing,
        labels=labels,
        onehot_enc=False,
        subset=config.data.subset,
        filter_meta=model_config.data.filter_meta,
        ram=config.data.ram,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,  # shuffle is not necessary
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
    device = f'cuda:{config.gpu_device}'
    model = model.to(device)
    model.eval()

    output_dir_real = Path(config.tmp_image_folder) / "fid/real"
    output_dir_fake = Path(config.tmp_image_folder) / "fid/fake"

    for odir in [output_dir_real, output_dir_fake]:
        odir.mkdir(exist_ok=True, parents=True)

    export_real_images(output_dir_real, train_dataloader)
    export_fake_images(
        model,
        output_dir_fake,
        num_images=len(dataset),
        batch_size=config.data.batch_size,
        device=device,
        latent_dim=model_config.latent_dim,
    )

    fid_score = fid.compute_fid(
        str(output_dir_real), 
        str(output_dir_fake), 
        batch_size=config.batch_size_fid_kid,
        device=torch.device(device),
        use_dataparallel=True,
    )
    kid_score = fid.compute_kid(
        str(output_dir_real), 
        str(output_dir_fake),
        batch_size=config.batch_size_fid_kid,
        device=torch.device(device),
        use_dataparallel=True,
    )
    print(f"fid: {fid_score}, kid: {kid_score}")

    shutil.rmtree(output_dir_real.parent)

    d = {"fid": [fid_score], "kid": [kid_score]}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(args.experiment_folder, "image_quality_metrics.csv"))
