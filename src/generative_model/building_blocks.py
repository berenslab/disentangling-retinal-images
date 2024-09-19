from typing import List
import numpy as np
import torch


@torch.jit.script
def clamp_gain(x: torch.Tensor, g: float, c: float):
    return torch.clamp(x * g, -c, c)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def identity(x):
    return x


def leaky_relu_0_2(x):
    return torch.nn.functional.leaky_relu(x, 0.2)


activation_funcs = {
    "linear": {"fn": identity, "def_gain": 1},
    "lrelu": {"fn": leaky_relu_0_2, "def_gain": np.sqrt(2)},
}


class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation="linear",
        lr_multiplier=1,
        bias_init=0,
    ):
        super().__init__()
        self.activation = activation_funcs[activation]["fn"]
        self.activation_gain = activation_funcs[activation]["def_gain"]
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain
        x = (
            self.activation(torch.addmm(b.unsqueeze(0), x, w.t()))
            * self.activation_gain
        )
        return x


class SmoothDownsample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = torch.nn.Parameter(kernel, requires_grad=False)
        self.pad = torch.nn.ReplicationPad2d((2, 1, 2, 1))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = torch.nn.functional.conv2d(x, self.kernel).view(b, c, h, w)
        x = torch.nn.functional.interpolate(
            x, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
        )
        return x


class SmoothUpsample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = torch.nn.Parameter(kernel, requires_grad=False)
        self.pad = torch.nn.ReplicationPad2d((2, 1, 2, 1))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.pad(x)
        x = torch.nn.functional.conv2d(x, self.kernel).view(b, c, h * 2, w * 2)
        return x


class EqualizedConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation="linear",
        resample=identity,
    ):
        super().__init__()
        self.resample = resample
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.activation = activation_funcs[activation]["fn"]
        self.activation_gain = activation_funcs[activation]["def_gain"]
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias[None, :, None, None] if self.bias is not None else 0
        x = self.resample(x)
        x = torch.nn.functional.conv2d(x, w, padding=self.padding)
        return clamp_gain(
            self.activation(x + b), self.activation_gain * gain, 256 * gain
        )


def modulated_conv2d(x, weight, styles, padding=0, demodulate=True):
    batch_size = x.shape[0]
    _, in_channels, kh, kw = weight.shape

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [NOIkk]
    w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = torch.nn.functional.conv2d(x, w, padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


class MappingNetwork(torch.nn.Module):
    """Mapping network for latent vector.

    Maps random vector to (style-disentangling) intermediate latent space.

    z_dim: Input latent (Z) dimensionality, 0 = no latent.
    w_dim: Intermediate latent (W) dimensionality.
    num_ws: Number of intermediate latents to output, None = do not broadcast.
    num_layers: Number of mapping layers.
    normalize: If true normalize z's second moment.
    activation: Activation function: 'relu', 'lrelu', etc.
    lr_multiplier: Learning rate multiplier for the mapping layers.
    w_avg_beta: Decay for tracking the moving average of W during training, None = do not track.
    """

    def __init__(
        self,
        z_dim,
        w_dim,
        num_ws,
        num_layers=8,
        normalize=False,
        activation="lrelu",
        lr_multiplier=0.01,
        w_avg_beta=0.995,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.normalize = normalize
        self.w_avg_beta = w_avg_beta

        features_list = [z_dim] + [w_dim] * num_layers

        self.layers = torch.nn.ModuleList()
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            self.layers.append(
                FullyConnectedLayer(
                    in_features,
                    out_features,
                    activation=activation,
                    lr_multiplier=lr_multiplier,
                )
            )

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim], dtype=torch.float16))

    def forward(
        self,
        z,
        truncation_psi=1,
        truncation_cutoff=None,
        skip_w_avg_update=False,
    ):
        # Embed, normalize.
        if self.normalize:
            z = normalize_2nd_moment(z)

        # Main layers.
        for idx in range(self.num_layers):
            z = self.layers[idx](z)

        # Update moving average of w.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(z.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            z = z.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                z = self.w_avg.lerp(z, truncation_psi)
            else:
                z[:, :truncation_cutoff] = self.w_avg.lerp(
                    z[:, :truncation_cutoff], truncation_psi
                )
        return z


class SeparateMappingNetwork(torch.nn.Module):
    """Mapping network for latent vector.

    Maps random vector to (style-disentangling) intermediate latent space.

    z_dim: Input latent (Z) dimensionality, 0 = no latent.
    w_dim: Intermediate latent (W) dimensionality.
    subspace_dims: List of subspace dimensions.
    num_ws: Number of intermediate latents to output, None = do not broadcast.
    num_layers: Number of mapping layers.
    activation: Activation function: 'relu', 'lrelu', etc.
    lr_multiplier: Learning rate multiplier for the mapping layers.
    w_avg_beta: Decay for tracking the moving average of W during training, None = do not track.
    """

    def __init__(
        self,
        z_dim: int,
        w_dim: int ,
        subspace_dims: List[int],
        num_ws: int,
        num_layers=8,
        normalize=False,
        activation="lrelu",
        lr_multiplier=0.01,
        w_avg_beta=0.995,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.subspace_dims = subspace_dims
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.normalize = normalize
        self.w_avg_beta = w_avg_beta
        sum_subspaces = sum(self.subspace_dims)

        self.w_mappings = torch.nn.ModuleList(
            [
                MappingNetwork(
                    z_dim=sub_w_dim,
                    w_dim=sub_w_dim,
                    num_ws=self.num_ws,
                    num_layers=num_layers,
                    normalize=True,
                    activation=activation,
                    lr_multiplier=lr_multiplier,
                    w_avg_beta=0.995,
                )
                for sub_w_dim in self.subspace_dims
            ]
        )
        self.w_mappings.append(
            MappingNetwork(
                z_dim=z_dim - sum_subspaces,
                w_dim=w_dim - sum_subspaces,
                num_ws=self.num_ws,
                num_layers=num_layers,
                normalize=True,
                w_avg_beta=0.995,
            )
        )

    def forward(
        self,
        z,
        truncation_psi=1,
        truncation_cutoff=None,
        skip_w_avg_update=False,
    ):
        subspace_cumsums = (
            [0] + list(np.cumsum(self.subspace_dims)) + [self.w_dim]
        )  # [0, 4, 16, 32]
        ws = torch.cat(
            [
                w_mapping(
                    z[:, start:end],
                    truncation_psi=truncation_psi,
                    truncation_cutoff=truncation_cutoff,
                    skip_w_avg_update=skip_w_avg_update,
                )
                for w_mapping, start, end in zip(
                    self.w_mappings, subspace_cumsums[:-1], subspace_cumsums[1:]
                )
            ],
            dim=-1,
        )

        return ws
