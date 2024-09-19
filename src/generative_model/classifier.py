import torch
import torchvision


class LinearClassifier(torch.nn.Module):
    """Linear classifier head.

    Attributes:
        w_shape: Number of latent space dimensions.
        c_shape: Number of classes.
    """

    def __init__(
        self,
        w_shape: int = 512,
        c_shape: int = 2,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(w_shape, c_shape)

    def forward(self, w):
        return self.linear(w)


class Classifier(torch.nn.Module):
    """MLP classifier (nonlinear).

    Attributes:
        hidden_layers: Number of hidden layers.
        w_shape: Number of latent space dimensions.
        c_shape: Number of classes.
    """

    def __init__(
        self,
        hidden_layers: int = 1,
        w_shape: int = 512,
        c_shape: int = 2,
    ):
        super().__init__()

        hidden_channels = [w_shape // (2 * (i + 1)) for i in range(hidden_layers)]

        self.classifier = torchvision.ops.MLP(
            in_channels=w_shape,
            hidden_channels=hidden_channels
            + [
                c_shape,
            ],
            activation_layer=torch.nn.ReLU,
        )

    def forward(self, w):
        return self.classifier(w)
