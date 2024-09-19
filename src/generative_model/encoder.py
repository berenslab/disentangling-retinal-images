import copy
import math
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import save_image


class Encoder(torch.nn.Module):
    """Image encoder with resnet backbone.

    Attributes:
        resnet_backbone: Resnet backbone architecture. One of 18, 34, 50.
        latent_dim: Output shape of encoder.
    """

    def __init__(
        self,
        resnet_backbone: int = 18,
        latent_dim: int = 512,
    ):
        super().__init__()

        if resnet_backbone == 18:
            backbone = torchvision.models.resnet18(weights=None)
        elif resnet_backbone == 34:
            backbone = torchvision.models.resnet34(weights=None)
        elif resnet_backbone == 50:
            backbone = torchvision.models.resnet50(weights=None)

        # Generate latent vector of the same dimension as z-dim.
        self.encoder = torch.nn.Sequential(
            backbone,
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1000, out_features=latent_dim),
        )

    def forward(self, batch):
        return self.encoder(batch)


class OfflineEncoder(pl.LightningModule):
    """Encoder trained on freezed GAN architecture.

    Invert image generation from latent space (based on random samples, unconditional).

    Attributes:
        generator: Generator of trained GAN.
        resnet_backbone: Resnet backbone architecture. One of 18, 34, 50.
            Pretrained weights on ImageNet are loaded.
        conditional: True if trained GAN is a conditional model.
        w_avg_samples: Number of samples for latent space (w-space) stats.
    """

    def __init__(
        self,
        generator,
        resnet_backbone: int = 18,
        conditional: bool = False,
        w_avg_samples: int = 10000,
    ):
        super().__init__()
        if resnet_backbone == 18:
            backbone = torchvision.models.resnet18(weights=None)
        elif resnet_backbone == 34:
            backbone = torchvision.models.resnet34(weights=None)
        elif resnet_backbone == 50:
            backbone = torchvision.models.resnet50(weights=None)

        self.generator = (
            copy.deepcopy(generator).eval().requires_grad_(False).to("cuda")
        )
        # Generate latent vector of the same dimension as z-dim.
        self.encoder = torch.nn.Sequential(
            backbone,
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1000, out_features=self.generator.w_dim),
        )
        self.output_dir_samples = self._create_image_directory()
        self.counter_val_steps = 0
        self.conditional = conditional

        # Compute w stats.
        z_samples = (
            np.random.RandomState(123)
            .randn(w_avg_samples, self.generator.z_dim)
            .astype(np.float32)
        )
        if self.conditional:
            w_samples = self.generator.wz_mapping(
                torch.from_numpy(z_samples).to("cuda")
            )  # [N, C]
        else:
            w_samples = self.generator.w_mapping(
                torch.from_numpy(z_samples).to("cuda")
            )  # [N, L, C]
            w_samples = w_samples[:, 0, :]  # [N, C]
        self.w_avg = torch.mean(w_samples, dim=0, keepdim=True)  # [1, C]
        # self.w_std = (torch.sum((w_samples - self.w_avg) ** 2) / w_avg_samples) ** 0.5

        # Load VGG16 feature detector.
        self.vgg16 = torchvision.models.vgg16(weights="VGG16_Weights")

    def forward(self, x):
        if type(x) is dict:
            latent = self.encoder(x["image"])
            latent += self.w_avg
            return latent  # for eval
        else:
            latent = self.encoder(x)
            latent += self.w_avg
            return latent

    def _shared_eval_step(self, batch, batch_idx):
        latent, labels = self(batch), None
        if self.conditional:
            labels = batch["labels"]

        target_images = batch["image"]

        # Features for target image.
        target_features = self._vgg_features(target_images)

        # Features for synth images.
        synth_images = self.generator.wz_to_image(latent, labels, noise_mode="const")
        synth_features = self._vgg_features(synth_images)

        perceptual_loss = (target_features - synth_features).square().sum()
        mse = torch.mean(
            (synth_images - target_images) ** 2
        )  # oom at validation (after 30%) if I use torch.nn.functional.mse_loss here

        loss = perceptual_loss + 10 * mse
        return loss, perceptual_loss, mse

    def _vgg_features(self, images):
        images = (images + 1) * (255 / 2)
        if images.shape[2] > 256:
            images = F.interpolate(images, size=(256, 256), mode="area")
        features = self.vgg16(images, resize_images=False, return_lpips=True)
        return features

    def training_step(self, batch, batch_idx):
        latent, labels = self(batch), None
        if self.conditional:
            labels = batch["labels"]

        target_images = batch["image"]
        # Features for target image.
        target_features = self._vgg_features(target_images)

        # Features for synth images.
        synth_images = self.generator.wz_to_image(latent, labels, noise_mode="const")
        synth_features = self._vgg_features(synth_images)

        perceptual_loss = (target_features - synth_features).square().sum()
        mse = torch.nn.functional.mse_loss(synth_images, target_images)

        loss = perceptual_loss + 10 * mse
        metrics = {
            "train_mse": mse,
            "train_perceptual_loss": perceptual_loss,
            "train_loss": loss,
        }
        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, perceptual_loss, mse = self._shared_eval_step(batch, batch_idx)
        metrics = {
            "val__mse": mse,
            "val_perceptual_loss": perceptual_loss,
            "val_loss": loss,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        batch = next(iter(self.trainer.val_dataloaders[0]))
        batch["image"] = batch["image"].to("cuda")
        latent, labels = self(batch), None
        if self.conditional:
            labels = batch["labels"].to("cuda")
        synth_images = self.generator.wz_to_image(latent, labels, noise_mode="const")
        if (
            self.current_epoch == 0
        ):  # no shuffle of validation data, target images will stay the same
            self._save_images(batch["image"], "reals")
        self._save_images(synth_images, "generated")

    def test_step(self, batch, batch_idx):
        loss, perceptual_loss, mse = self._shared_eval_step(batch, batch_idx)
        metrics = {"mse": mse, "perceptual_loss": perceptual_loss, "val_loss": loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.encoder.parameters(), lr=0.0001)

    def _save_images(self, images, name: str = "generated"):
        save_image(
            images,
            self.output_dir_samples / f"{name}_{self.global_step:06d}.png",
            nrow=int(math.sqrt(images.shape[0])),
            value_range=(-1, 1),
            normalize=True,
        )

    def _create_image_directory(self):
        output_dir_samples = Path(os.getcwd()) / "images/"
        output_dir_samples.mkdir(exist_ok=True, parents=True)
        return output_dir_samples
