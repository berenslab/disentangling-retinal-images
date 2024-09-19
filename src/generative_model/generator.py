from typing import List
import numpy as np
import torch

from src.generative_model.building_blocks import (FullyConnectedLayer,
                                                  MappingNetwork,
                                                  SeparateMappingNetwork,
                                                  SmoothUpsample, activation_funcs,
                                                  clamp_gain, identity,
                                                  modulated_conv2d)


class Generator(torch.nn.Module):
    """Generator of StyleGAN2.

    Attributes:
        z_dim: Input latent (Z) dimensionality.
        c_dim: Conditioning label (C) dimensionality.
        w_dim: Intermediate latent (W) dimensionality.
        subspace_dims: List of subspace dimensions.
        seperate_mapping_networks:
        w_num_layers: Number of mapping layers.
        img_resolution: Output resolution.
        img_channels: Number of output color channels.
        synthesis_layer: Kind of synthesis layer, either stylegan1 or stylegan2.
    """

    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        w_dim: int,
        subspace_dims: List[int],
        seperate_mapping_networks: bool,
        w_num_layers: int,
        img_resolution: int,
        img_channels: int,
        synthesis_layer: str = "stylegan2",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.subspace_dims = subspace_dims
        self.seperate_mapping_networks = seperate_mapping_networks
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            synthesis_layer=synthesis_layer,
        )
        self.num_ws = self.synthesis.num_ws
        if c_dim > 0:
            self.wz_mapping = MappingNetwork(
                z_dim=z_dim,
                w_dim=w_dim,
                num_ws=None,
                num_layers=w_num_layers,
                normalize=True,
                w_avg_beta=None,
            )
            self.wc_mapping = MappingNetwork(
                z_dim=c_dim,
                w_dim=w_dim,
                num_ws=None,
                num_layers=w_num_layers,
                normalize=True,
                w_avg_beta=None,
            )
        if self.seperate_mapping_networks:
            self.w_mapping = SeparateMappingNetwork(
                z_dim=w_dim * 2 if c_dim > 0 else z_dim,
                w_dim=w_dim,
                subspace_dims=subspace_dims,
                num_ws=self.num_ws,
                num_layers=2 if c_dim > 0 else w_num_layers,
                normalize=False if c_dim > 0 else True,
                w_avg_beta=0.995,
            )
        else:
            self.w_mapping = MappingNetwork(
                z_dim=w_dim * 2 if c_dim > 0 else z_dim,
                w_dim=w_dim,
                num_ws=self.num_ws,
                num_layers=2 if c_dim > 0 else w_num_layers,
                normalize=False if c_dim > 0 else True,
                w_avg_beta=0.995,
            )

    def forward(
        self, z, c=None, truncation_psi=1, truncation_cutoff=None, noise_mode="random"
    ):
        if c is not None:
            wz = self.wz_mapping(z)
            wc = self.wc_mapping(c)
            z = torch.cat([wz, wc], dim=1)
        ws = self.w_mapping(
            z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
        )
        img = self.synthesis(ws, noise_mode)
        return img

    def wz_to_image(
        self, wz, c=None, truncation_psi=1, truncation_cutoff=None, noise_mode="const"
    ):
        if c is not None:
            wc = self.wc_mapping(c)
            z = torch.cat([wz, wc], dim=1)
            ws = self.w_mapping(
                z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
            )
        else:
            if self.num_ws is not None:
                # Note: truncation cannot be applied, but maybe this is only done during training anyhow.
                ws = wz.unsqueeze(1).repeat([1, self.num_ws, 1])
            else:
                ws = wz
        img = self.synthesis(ws, noise_mode)
        return img


class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim: int,
        img_resolution: int,
        img_channels: int,
        channel_base: int = 16384,
        channel_max: int = 512,
        synthesis_layer: str = "stylegan2",
    ):
        super().__init__()

        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2**i for i in range(2, self.img_resolution_log2 + 1)]
        self.num_ws = 2 * (len(self.block_resolutions) + 1)
        channels_dict = {
            res: min(channel_base // res, channel_max) for res in self.block_resolutions
        }
        self.blocks = torch.nn.ModuleList()
        self.first_block = SynthesisPrologue(
            channels_dict[self.block_resolutions[0]],
            w_dim=w_dim,
            resolution=self.block_resolutions[0],
            img_channels=img_channels,
            synthesis_layer=synthesis_layer,
        )
        for res in self.block_resolutions[1:]:
            in_channels = channels_dict[res // 2]
            out_channels = channels_dict[res]
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                synthesis_layer=synthesis_layer,
            )
            self.blocks.append(block)

    def forward(self, ws, noise_mode="random"):
        split_ws = [ws[:, 0:2, :]] + [
            ws[:, 2 * n + 1 : 2 * n + 4, :] for n in range(len(self.block_resolutions))
        ]
        x, img = self.first_block(split_ws[0], noise_mode)
        for i in range(len(self.block_resolutions) - 1):
            x, img = self.blocks[i](x, img, split_ws[i + 1], noise_mode)
        return img


class SynthesisPrologue(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        w_dim: int,
        resolution: int,
        img_channels: int,
        synthesis_layer: str,
    ):
        super().__init__()
        SynthesisLayer = (
            SynthesisLayer2 if synthesis_layer == "stylegan2" else SynthesisLayer1
        )
        ToRGBLayer = ToRGBLayer2 if synthesis_layer == "stylegan2" else ToRGBLayer1
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.const = torch.nn.Parameter(
            torch.randn([out_channels, resolution, resolution])
        )
        self.conv1 = SynthesisLayer(
            out_channels, out_channels, w_dim=w_dim, resolution=resolution
        )
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)

    def forward(self, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))
        x = self.const.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)
        img = self.torgb(x, next(w_iter))
        return x, img


class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        resolution: int,
        img_channels: int,
        synthesis_layer: str,
    ):
        super().__init__()
        SynthesisLayer = (
            SynthesisLayer2 if synthesis_layer == "stylegan2" else SynthesisLayer1
        )
        ToRGBLayer = ToRGBLayer2 if synthesis_layer == "stylegan2" else ToRGBLayer1
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.resampler = SmoothUpsample()
        self.conv0 = SynthesisLayer(
            in_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            resampler=self.resampler,
        )
        self.conv1 = SynthesisLayer(
            out_channels, out_channels, w_dim=w_dim, resolution=resolution
        )
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)

    def forward(self, x, img, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))

        x = self.conv0(x, next(w_iter), noise_mode=noise_mode)
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)

        y = self.torgb(x, next(w_iter))
        img = self.resampler(img)
        img = img.add_(y)

        return x, img


class ToRGBLayer2(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, w_dim: int, kernel_size: int = 1
    ):
        super().__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False)
        return torch.clamp(x + self.bias[None, :, None, None], -256, 256)


class ToRGBLayer1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, _w):
        # note that StyleGAN1's rgb layer doesnt use any style
        w = self.weight * self.weight_gain
        x = torch.nn.functional.conv2d(x, w)
        return torch.clamp(x + self.bias[None, :, None, None], -256, 256)


class SynthesisLayer2(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        kernel_size=3,
        resampler=identity,
        activation="lrelu",
    ):
        super().__init__()
        self.resolution = resolution
        self.resampler = resampler
        self.activation = activation_funcs[activation]["fn"]
        self.activation_gain = activation_funcs[activation]["def_gain"]
        self.padding = kernel_size // 2
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        )

        self.register_buffer("noise_const", torch.randn([resolution, resolution]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == "random":
            noise = (
                torch.randn(
                    [x.shape[0], 1, self.resolution, self.resolution], device=x.device
                )
                * self.noise_strength
            )
        if noise_mode == "const":
            noise = self.noise_const * self.noise_strength

        x = modulated_conv2d(
            x=x, weight=self.weight, styles=styles, padding=self.padding
        )
        x = self.resampler(x)
        x = x + noise

        return clamp_gain(
            self.activation(x + self.bias[None, :, None, None]),
            self.activation_gain * gain,
            256 * gain,
        )


class SynthesisLayer1(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        kernel_size=3,
        resampler=identity,
        activation="lrelu",
    ):
        super().__init__()
        self.resolution = resolution
        self.resampler = resampler
        self.activation = activation_funcs[activation]["fn"]
        self.activation_gain = activation_funcs[activation]["def_gain"]
        self.padding = kernel_size // 2
        self.affine = FullyConnectedLayer(w_dim, out_channels * 2, bias_init=1)
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        )
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.ada_in = AdaIN(out_channels)
        self.register_buffer("noise_const", torch.randn([resolution, resolution]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == "random":
            noise = (
                torch.randn(
                    [x.shape[0], 1, self.resolution, self.resolution], device=x.device
                )
                * self.noise_strength
            )
        if noise_mode == "const":
            noise = self.noise_const * self.noise_strength

        w = self.weight * self.weight_gain
        x = torch.nn.functional.conv2d(x, w, padding=self.padding)

        x = self.resampler(x)
        x = x + noise
        x = clamp_gain(
            self.activation(x + self.bias[None, :, None, None]),
            self.activation_gain * gain,
            256 * gain,
        )
        x = self.ada_in(x, styles)
        return x


class AdaIN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(in_channels)

    def forward(self, x, style):
        style = style.unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(x)
        out = gamma * out + beta

        return out
