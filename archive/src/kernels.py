import torch
import autograd.numpy as np


def l2norm(X, Y):
    """Compute \|X - Y\|_2^2 of tensors X, Y
    """
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
    return dnorm2

class RBF(torch.nn.Module):
    """For GSVGD to work, a kernel class need to have the following methods:
        forward: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma=None, method="med_heuristic"):
        super().__init__()
        self.sigma = sigma
        self.method = method

    def forward(self, X, Y):
        dnorm2 = l2norm(X, Y)

        # median heuristic
        if self.method == "med_heuristic" or self.sigma is None:
            sigma2 = torch.median(dnorm2) / (2 * np.log(X.size(0)))
            # sigma2 = torch.median(dnorm2) / 2
            self.sigma = sigma2.sqrt().item()

        # 1/sigma^2
        two_sigma2_inv = 1.0 / (1e-8 + 2 * self.sigma ** 2)
        K_XY = (-two_sigma2_inv * dnorm2).exp()

        return K_XY

    def grad_first(self, X, Y):
        """Compute grad_K in wrt first argument in matrix form 
        """
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        """Compute grad_K in wrt second argument in matrix form 
        """
        # (symmetric kernel)
        # Gram matrix
        sigma2_inv = 1 / (1e-8 + self.sigma ** 2)
        # TODO: Is this correct?
        a = (-l2norm(X, Y) * 0.5 * sigma2_inv).exp().unsqueeze(0)
        # [diff]_{ijk} = y^i_j - x^i_k
        diff = (Y.unsqueeze(1) - X).transpose(0, -1)
        # compute grad_K
        grad_K_XY = -sigma2_inv * diff * a

        return grad_K_XY

    def gradgrad(self, X, Y):
        """
        Args:
            X: torch.Tensor of shape (n, dim)
            Y: torch.Tensor of shape (m, dim)
        Output:
            torch.Tensor of shape (dim, dim, n, m)
        """
        # Gram matrix
        sigma2_inv = 1 / (1e-8 + self.sigma ** 2)
        a = (-l2norm(X, Y) * 0.5 * sigma2_inv).exp().unsqueeze(0)
        # [diff]_{ijk} = y^i_j - x^i_k
        diff = (X - Y.unsqueeze(1)).transpose(0, -1)
        # product of differences
        diff_outerprod = -diff.unsqueeze(0) * diff.unsqueeze(1)
        # compute gradgrad_K
        # TODO: How do we avoid doing to(device)??
        diag = sigma2_inv * torch.eye(X.shape[1], device=X.device).unsqueeze(
            -1
        ).unsqueeze(-1)
        gradgrad_K_all = (diag + sigma2_inv ** 2 * diff_outerprod) * a
        return gradgrad_K_all

