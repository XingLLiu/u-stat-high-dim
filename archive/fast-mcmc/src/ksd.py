import torch
import torch.autograd as autograd
import autograd.numpy as np
from tqdm import tqdm

class KSD:
    def __init__(
        self,
        target: torch.distributions.Distribution,
        kernel: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device="cpu",
    ):
        """
        Args:
            target (torch.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
            kernel (torch.nn.Module): [description]
            optimizer (torch.optim.Optimizer): [description]
        """
        self.p = target
        self.k = kernel
        self.optim = optimizer
        self.device = device

    def phi(self, X: torch.Tensor, **kwargs):
        """

        Args:
            X (torch.Tensor): Particles being transported to the target distribution

        Returns:
            phi (torch.Tensor): Functional gradient
        """
        # copy the data for X into X
        X = X.detach().requires_grad_(True)

        log_prob = self.p.log_prob(X, **kwargs)
        # print(X)
        # print(log_prob[:5])
        score_func = autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.k(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
        return phi

    def step(self, X: torch.Tensor, **kwargs):
        """Gradient descent step

        Args:
            X (torch.Tensor): Particles to transport to the target distribution
        """
        self.optim.zero_grad()
        phi = self.phi(X, **kwargs)
        X.grad = -phi
        self.optim.step()

        # particle-averaged magnitude
        # pam = torch.linalg.norm(phi.detach(), dim=1).mean()
        pam = torch.max(phi.detach().abs(), dim=1)[0].mean()
        return pam.item()

    def fit(self, x0: torch.Tensor, epochs: torch.int64, verbose: bool = True,
        metric: callable = None,
        save_every: int = 100,
        threshold: float = 0
    ):
        """
        Args:
            x0 (torch.Tensor): Initial set of particles to be updated
            epochs (torch.int64): Number of gradient descent iterations
        """
        self.metrics = [0] * (epochs//save_every)
        self.particles = [0] * (1 + epochs//save_every)
        self.particles[0] = x0.clone().detach().cpu()
        self.pam = [0] * (epochs//save_every)

        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        
        for i in iterator:
            pam = self.step(x0)
            if (i+1) % save_every == 0:
                # self.metrics[i//save_every] = metric(x0.detach())
                self.particles[1 + i//save_every] = x0.clone().detach().cpu()
                # self.particles[1 + i//save_every] = x0.clone().detach()
                self.pam[i//save_every] = pam

            # early stop
            if pam < threshold:
                print(f"GSVGD converged in {i+1} epochs as PAM {pam} is less than {threshold}")
                break
        
        return self.metrics

