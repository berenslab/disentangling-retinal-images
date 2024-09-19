import numpy as np
import torch

from src.generative_model.building_blocks import (EqualizedConv2d,
                                                  FullyConnectedLayer,
                                                  MappingNetwork, SmoothDownsample)


class Discriminator(torch.nn.Module):
    """Progessiveley growing discriminator from the ProGAN paper.

    Here, we extend the discriminator as an encoder (latent_dim attribute).

    Reference: https://arxiv.org/abs/1710.10196

    Attributes:
        c_dim: Conditioning label (C) dimensionality.
        img_resolution: Input image resolution.
        img_channels: Number of input color channels.
        channel_base: Overall multiplier for the number of channels.
        channel_max: Maximum number of channels in any layer.
        w_num_layers: Number of mapping layers for conditional labels.
        latent_dim: Dimension of latent space prediction (extends the discriminator as an encoder).
    """

    def __init__(
        self,
        c_dim,
        img_resolution,
        img_channels,
        channel_base=16384,
        channel_max=512,
        w_num_layers=None,
        latent_dim: int = 512,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(self.img_resolution_log2, 2, -1)
        ]  # img_resolution 256: [256, 128, 64, 32, 16, 8]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions + [4]
        }  # img_resolution 256: {256: 64, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512}
        if c_dim == 0:
            cmap_dim = 0
        else:
            cmap_dim = channels_dict[4]

        module_list = [
            EqualizedConv2d(
                img_channels,
                channels_dict[img_resolution],
                kernel_size=1,
                activation="lrelu",
            )
        ]
        for res in self.block_resolutions:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            module_list.append(DiscriminatorBlock(in_channels, out_channels))
        if c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=c_dim,
                w_dim=cmap_dim,
                num_ws=None,
                w_avg_beta=None,
                num_layers=w_num_layers,
            )
        self.net = torch.nn.Sequential(*module_list)
        self.b4 = DiscriminatorEpilogue(
            channels_dict[4], cmap_dim=cmap_dim, resolution=4, latent_dim=latent_dim
        )

    def forward(self, x, c=None):
        x = self.net(x)
        if self.c_dim > 0:
            cmap = self.mapping(c)
        else:
            cmap = None
        decision, w_hat, features = self.b4(x, cmap)
        return decision, w_hat, features


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation="lrelu"):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = 0
        downsampler = SmoothDownsample()
        self.conv0 = EqualizedConv2d(
            in_channels, in_channels, kernel_size=3, activation=activation
        )
        self.conv1 = EqualizedConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            resample=downsampler,
        )
        self.skip = EqualizedConv2d(
            in_channels, out_channels, kernel_size=1, bias=False, resample=downsampler
        )

    def forward(self, x):
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        x = y.add_(x)
        return x


class DiscriminatorEpilogue(torch.nn.Module):
    """Discriminator epilogue.

    Attributes:
        in_channels: Number of input channels.
        cmap_dim: Dimensionality of mapped conditioning label, 0 = no label.
        resolution: Resolution of this block.
        mbstd_group_size: Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels: Number of features for the minibatch standard deviation layer, 0 = disable.
        activation: Activation function: 'relu', 'lrelu', etc.
        latent_dim: Dimension of latent space prediction (extends the discriminator as an encoder).
    """

    def __init__(
        self,
        in_channels,
        cmap_dim,
        resolution,
        mbstd_group_size=None,
        mbstd_num_channels=1,
        activation="lrelu",
        latent_dim: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution

        self.mbstd = (
            MinibatchStdLayer(
                group_size=mbstd_group_size, num_channels=mbstd_num_channels
            )
            if mbstd_num_channels > 0
            else None
        )
        self.conv = EqualizedConv2d(
            in_channels + mbstd_num_channels,
            in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.fc = FullyConnectedLayer(
            in_channels * (resolution**2), in_channels, activation=activation
        )
        self.discriminator_out = FullyConnectedLayer(
            in_channels, 1 if cmap_dim == 0 else cmap_dim
        )
        self.encoder_out = FullyConnectedLayer(in_channels, latent_dim)

    def forward(self, x, cmap):
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        discr_out = self.discriminator_out(x)
        enc_out = self.encoder_out(x)

        # Conditioning with projection discriminator (https://arxiv.org/pdf/1802.05637.pdf).
        if self.cmap_dim > 0:
            discr_out = (discr_out * cmap).sum(dim=1, keepdim=True) * (
                1 / np.sqrt(self.cmap_dim)
            )

        return discr_out, enc_out, x


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = (
            torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
            if self.group_size is not None
            else N
        )
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW] Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW] Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF] Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11] Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW] Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW] Append to input as new channels.
        return x
