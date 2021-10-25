
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
    X_cp = X.clone().detach().requires_grad_()
    Y = X.clone().detach()

    log_prob = self.p.log_prob(X_cp, **kwargs)
    score_func = autograd.grad(log_prob.sum(), X_cp)[0]

    X_cp = X.clone().detach().requires_grad_()
    with torch.no_grad():
        self.k.bandwidth(X, X)
    K_XX = self.k(X_cp, Y)
    grad_K = -autograd.grad(K_XX.sum(), X_cp)[0]

    # compute update rule
    attraction = K_XX.detach().matmul(score_func) / X.size(0)
    repulsion = grad_K / X.size(0)
    phi = attraction + repulsion

    return phi, repulsion

  def __call__(self, X: torch.Tensor, Y: torch.Tensor):
    """
    Args:
      X: n x dim
      Y: m x dim
    """
    assert not X.requires_grad_
    assert not Y.requires_grad_

    # copy data for score computation
    X_cp = X.clone().requires_grad_()
    Y_cp = Y.clone().requires_grad_()

    log_prob_X = self.p.log_prob(X_cp)
    log_prob_Y = self.p.log_prob(Y_cp)
    score_X = autograd.grad(log_prob_X.sum(), X_cp)[0] # n x dim
    score_Y = autograd.grad(log_prob_Y.sum(), Y_cp)[0] # m x dim

    # median heuristic
    X_cp = X.clone().detach().requires_grad_()
    with torch.no_grad():
      self.k.bandwidth(X, Y)
    
    # kernel
    K_XY = self.k(X, Y) # n x m
    
    # kernel grad
    grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
    grad_K_X = self.k.grad_first(X, Y) # n x m x dim

    # term 1
    term1 = score_X.unsqueeze(1) * score_Y.unsqueeze(0) * K_XY.unsqueeze(2) # n x m x dim
    # term 2
    term2 = score_X.unsqueeze(1) * grad_K_Y # n x m x dim
    # term3
    term3 = score_Y.unsqueeze(0) * grad_K_X # n x m x dim
    # term4
    term4 = self.k.gradgrad(X, Y) # n x m x dim x dim

    ksd = torch.sum(term1 + term2 + term3 + term4) / (X.shape[0] * Y.shape[0])
    return ksd