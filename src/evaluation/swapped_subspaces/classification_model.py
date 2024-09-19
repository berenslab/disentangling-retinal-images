import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import OmegaConf

from src.generative_model.encoder import Encoder
from src.generative_model.stylegan import StyleGAN2Model
from src.utils.metrics import SimpleMetric


class ClassificationModel(pl.LightningModule):
    """Image classification model.

    Attributes:
        resnet_backbone: Resnet backbone architecture for image encoder.
            One of 18, 34, 50.
        weight_decay: Adam optimizer weight decay. Defaults to 0.
        num_classes: Number of classes.
        gan_model_exp_folder: Optional experiment folder to generative model
            that maps images into their reconstruction space.
    """

    def __init__(
        self,
        resnet_backbone: int = 18,
        weight_decay: float = 0.0,
        num_classes: int = 3,
        gan_model_exp_folder: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = Encoder(
            resnet_backbone,
            latent_dim=num_classes,
        )
        self.weight_decay = weight_decay

        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.classification.MulticlassAccuracy(
                    num_classes, average="micro"
                ),
                torchmetrics.classification.MulticlassJaccardIndex(
                    num_classes, average="macro"
                ),
            ]
        )
        self.metrics_train = metrics.clone(prefix="train_")
        self.train_loss = SimpleMetric()
        self.metrics_val = metrics.clone(prefix="val_")
        self.val_loss = SimpleMetric()
        self.metrics_test = metrics.clone(prefix="test_")
        self.metrics_test_swapped = metrics.clone(prefix="test_swapped_")
        self.metrics_test_swapped_old_labels = metrics.clone(
            prefix="test_swapped_old_labels_"
        )

        if gan_model_exp_folder is not None:
            model_config = OmegaConf.load(
                os.path.join(gan_model_exp_folder, "config.yaml")
            )
            self.reconstruction_model = StyleGAN2Model(
                config=model_config,
                experiment_folder=gan_model_exp_folder,
                lambda_gp=0,
                class_dims=[3, 14],
            )
            with open(os.path.join(gan_model_exp_folder, "best_ckpt.txt"), "r") as file:
                checkpoint_file = file.read().rstrip()
            checkpoint = torch.load(checkpoint_file)
            self.reconstruction_model.load_state_dict(checkpoint["state_dict"])
            self.reconstruction_model.eval()
        else:
            self.reconstruction_model = None

    def forward(self, batch):
        image = batch["image"]
        return self.encoder(image)

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            weight_decay=self.weight_decay,
        )
        return optim

    def _shared_eval_step(self, batch, batch_idx, state: str) -> float:
        image, y = batch["image"], batch["labels"]
        y = y.squeeze()
        if self.reconstruction_model is not None:
            _, w_real_hat, _ = self.reconstruction_model.D(image, None)
            image = self.reconstruction_model.G.wz_to_image(
                wz=w_real_hat,
                c=None,
            )
        y_hat = self.encoder(image)
        loss = torch.nn.functional.cross_entropy(
            target=y,
            input=y_hat,
        )

        if state == "train":
            self.train_loss.update(loss)
            self.metrics_train.update(y_hat, y)
        elif state == "val":
            self.val_loss.update(loss)
            self.metrics_val.update(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx) -> float:
        return self._shared_eval_step(batch, batch_idx, state="train")

    def validation_step(self, batch, batch_idx) -> float:
        self._shared_eval_step(batch, batch_idx, state="val")

    def test_step(self, batch, batch_idx) -> float:
        image, y = batch["image"], batch["labels"]
        y = y.squeeze()
        if self.reconstruction_model is not None:
            _, w_real_hat, _ = self.reconstruction_model.D(image, None)
            # swap the encodings' age subspaces...
            # ...and the labels in the same manner
            w_real_hat_swapped = torch.empty_like(w_real_hat)
            y_swapped = torch.empty_like(y)

            batchsize = image.shape[0]
            for i in range(batchsize):
                if i < (batchsize - 1):
                    w_real_hat_swapped[i, :4] = w_real_hat[i + 1, :4]
                    w_real_hat_swapped[i, 4:] = w_real_hat[i, 4:]
                    y_swapped[i] = y[i + 1]
                else:
                    w_real_hat_swapped[i, :4] = w_real_hat[0, :4]
                    w_real_hat_swapped[i, 4:] = w_real_hat[i, 4:]
                    y_swapped[i] = y[0]

            image = self.reconstruction_model.G.wz_to_image(
                wz=w_real_hat,
                c=None,
            )
            image_swapped = self.reconstruction_model.G.wz_to_image(
                wz=w_real_hat_swapped,
                c=None,
            )
        y_hat = self.encoder(image)
        y_hat_swapped = self.encoder(image_swapped)
        self.metrics_test.update(y_hat, y)
        self.metrics_test_swapped.update(y_hat_swapped, y_swapped)
        self.metrics_test_swapped_old_labels.update(y_hat_swapped, y)

    def on_train_epoch_end(self) -> None:
        metric_dict = {
            "train_loss": self.train_loss.compute(),
            "step": float(self.current_epoch),
        }
        metric_dict.update(self.metrics_train.compute())

        self.log_dict(
            metric_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.train_loss.reset()
        self.metrics_train.reset()

    def on_validation_epoch_end(self) -> None:
        metric_dict = {
            "val_loss": self.val_loss.compute(),
            "step": float(self.current_epoch),
        }
        metric_dict.update(self.metrics_val.compute())

        self.log_dict(
            metric_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.val_loss.reset()
        self.metrics_val.reset()

    def on_test_epoch_end(self) -> None:
        metric_dict = {
            "step": float(self.current_epoch),
        }
        metric_dict.update(self.metrics_test.compute())
        metric_dict.update(self.metrics_test_swapped.compute())
        metric_dict.update(self.metrics_test_swapped_old_labels.compute())

        self.log_dict(
            metric_dict,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.metrics_test.reset()
        self.metrics_test_swapped.reset()
        self.metrics_test_swapped_old_labels.reset()
