import numpy as np
import torch
from torch import nn


def compute_gradient_penalty(x, d):
    gradients = torch.autograd.grad(
        outputs=[d.sum()], inputs=[x], create_graph=True, only_inputs=True
    )[0]
    r1_penalty = gradients.square().sum([1, 2, 3]).mean()
    return r1_penalty


class PathLengthPenalty(nn.Module):
    def __init__(self, pl_decay, pl_batch_shrink):
        super().__init__()
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_mean = nn.Parameter(torch.zeros([1]), requires_grad=False)

    def forward(self, fake, w):
        pl_noise = torch.randn_like(fake) / np.sqrt(fake.shape[2] * fake.shape[3])
        pl_grads = torch.autograd.grad(
            outputs=[(fake * pl_noise).sum()],
            inputs=[w],
            create_graph=True,
            only_inputs=True,
        )[0]
        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        return pl_penalty.mean()


def distance_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
):
    """Calculate the empirical distance correlation for multi-GPU DDP setting as
    described in [2].

    This statistic describes the dependence between `x` and `y`, which are
    random vectors of arbitrary length. The statistics' values range between 0
    (implies independence) and 1 (implies complete dependence).

    Args:
        x: Tensor of shape (batch-size, x_dimensions).
        y: Tensor of shape (batch-size, y_dimensions).

    Returns:
        The empirical distance correlation between `x` and `y`

    References:
        [1] https://en.wikipedia.org/wiki/Distance_correlation
        [2] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
            "Measuring and testing dependence by correlation of distances".
            Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.
    """
    # Euclidean distance between vectors.
    a = torch.cdist(x, x, p=2)  # batch-size x batch-size
    b = torch.cdist(y, y, p=2)  # batch-size x batch-size

    a_row_means = a.mean(axis=0, keepdims=True)
    b_row_means = b.mean(axis=0, keepdims=True)
    a_col_means = a.mean(axis=1, keepdims=True)
    b_col_means = b.mean(axis=1, keepdims=True)
    a_mean = a.mean()
    b_mean = b.mean()

    # Empirical distance matrices.
    A = a - a_row_means - a_col_means + a_mean
    B = b - b_row_means - b_col_means + b_mean

    # Empirical distance covariance.
    dcov = torch.mean(A * B)

    # Empirical distance variances.
    dvar_x = torch.mean(A * A)
    dvar_y = torch.mean(B * B)

    dCor = torch.sqrt(dcov / torch.sqrt(dvar_x * dvar_y))
    return dCor


def gaussian_mutual_information(x: torch.Tensor, y: torch.Tensor):
    """Compute mutual information between tensor x and tensor y.

    For this approximation we assume that x and y are random vectors
    following a multi-variate normal distribution.

    Following this assumption the mutual information is
    I(x,y) = 0.5 * log( ( det(cov(x)) det(cov(y)) ) / det(cov([x,y])) )

    For the exact derivation, see
    https://stats.stackexchange.com/questions/438607/mutual-information-between-subsets-of-variables-in-the-multivariate-normal-distr

    Args:
        x: Tensor of shape (batch-size, x_dimensions).
        y: Tensor of shape (batch-size, y_dimensions).
    """
    batch_size, x_dim = x.shape

    xy = torch.cat([x, y], dim=1)  # (batch-size, x_dim + y_dim)
    xy_mean = torch.mean(xy, dim=0, keepdims=True)  # (1, x_dim + y_dim)
    xy_centered = xy - xy_mean

    # Estimate covariance matrices from batch samples.
    cov_xy = (
        1 / (batch_size - 1) * xy_centered.T @ xy_centered
    )  # (x_dim + y_dim, x_dim + y_dim)

    # numerically stabilize
    cov_xy = cov_xy + 1e-6 * torch.eye(xy.shape[1], device=x.device)

    cov_x = cov_xy[:x_dim, :x_dim]
    cov_y = cov_xy[x_dim:, x_dim:]
    # loss = 0.5 * torch.log((torch.det(cov_x) * torch.det(cov_y)) / torch.det(cov_xy))
    loss = 0.5 * (
        torch.log(torch.det(cov_x))
        + torch.log(torch.det(cov_y))
        - torch.log(torch.det(cov_xy))
    )
    return torch.nan_to_num(loss, nan=0.0, posinf=0, neginf=0)


def gaussian_mutual_information_sum(x: torch.Tensor, y: torch.Tensor):
    """Compute mutual information between tensor x and tensor y.

    For this approximation we assume that x and y are random vectors
    following a multi-variate normal distribution.

    Following this assumption the mutual information is
    I(x,y) = 0.5 log( ( det(cov(x)) det(cov(y)) ) / det(cov([x,y])) )

    For the exact derivation, see
    https://stats.stackexchange.com/questions/438607/mutual-information-between-subsets-of-variables-in-the-multivariate-normal-distr

    To make things simpler use the sum of the variables x and y.
    Mutual information calculation changes to
    I(x,y) = 0.5 * log( var(x_sum) * var(y_sum) / (var(x_sum) * var(y_sum) - cov(x_sum, y_sum)**2) )

    Args:
        x: Tensor of shape (batch-size, x_dimensions).
        y: Tensor of shape (batch-size, y_dimensions).
    """
    batch_size = x.shape[0]

    x_sum = torch.sum(x, dim=1)
    y_sum = torch.sum(y, dim=1)

    x_sum_mean = torch.mean(x_sum)
    y_sum_mean = torch.mean(y_sum)

    cov_sums = (
        (x_sum - x_sum_mean).unsqueeze(0)
        @ (y_sum - y_sum_mean).unsqueeze(1)
        / (batch_size - 1)
    )
    cov_sums = cov_sums.squeeze()

    numerator = torch.var(x_sum) * torch.var(y_sum)
    loss = 0.5 * (torch.log(numerator) - torch.log(numerator - cov_sums**2))
    return torch.nan_to_num(loss, nan=0.0, posinf=0, neginf=0)
