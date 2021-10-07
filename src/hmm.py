import tensorflow as tf
import numpy as np

class LinearHMM:
  def __init__(self, A, B, C, D, Q, R, P, N=1):
    """
      shape of latent vars = (n_step, n_particles, dim_x)
      shape of obs = (n_step, n_particles, dim_y)
      Q: var of noise to latent vars
      R: var of noise to observations
      P: var of x_0
    """
    self.N = N
    self.A = A # dx x dx
    self.B = B # dx x dv
    self.C = C # dy x dx
    self.D = D # dy x dw
    self.Q = Q # dv x dv, symmetric
    self.R = R # dw x dw, symmetric
    self.P = P # dx x dx, symmetric
    # self.var_v = B @ Q @ B.T
    # self.var_w = D @ R @ D.T
    self.v_mean = np.zeros(B.shape[1])
    self.w_mean = np.zeros(D.shape[1])
    self.x0_mean = np.zeros(P.shape[1])
    self.latent = []
    self.obs = []

    self.dim_x = A.shape[1]
    self.dim_v = B.shape[1]
    self.dim_w = D.shape[1]

  def initialize_latent(self):
    """initialize x0"""
    x0 = np.random.multivariate_normal(self.x0_mean, self.P, (1, self.N)) # 1 x N x dx
    self.latent = x0
  
  def simulate_latent(self):
    """simulate 1 more latent vars"""
    v_n = np.random.multivariate_normal(self.v_mean, self.Q, (1, self.N)) # 1 x N x dv
    x_n = self.latent[-1] @ self.A.T + v_n @ self.B.T # 1 x N x dx
    # self.latent.append(x_n)
    self.latent = np.concatenate((self.latent, x_n), axis=0)
    
  def simulate_obs(self):
    """simulate 1 more observation"""
    if len(self.latent) >= 2:  
      n = len(self.obs) + 1 # index to simulate; starting from 1
      w_n = np.random.multivariate_normal(self.w_mean, self.R, (1, self.N)) # 1 x N x dw
      y_n = self.latent[n] @ self.C.T + w_n @ self.D.T # 1 x N x dy
      # self.obs.append(y_n)
      if len(self.obs) == 0:
        self.obs = y_n
      else:
        self.obs = np.concatenate((self.obs, y_n), axis=0)

  def simulate(self, steps):
    """simulate 'steps' more latent vars and obs"""
    if len(self.latent) == 0:
      self.initialize_latent()

    for i in range(steps):
      self.simulate_latent()
      self.simulate_obs()

  def log_prob(self, steps=None):
    """*unnormalized* log prob of x_{1:steps}
    Input:
      steps: must be <= n
    Output:
      gamma: N
    """
    steps = self.latent.shape[0]-1 if steps is None else steps

    log_gamma = self.log_prob_cond(0)
    for i in range(steps):
      new_log_prob = self.log_prob_cond(i+1)
      log_gamma = log_gamma + new_log_prob
    return log_gamma

  def log_prob_cond(self, n_step):
    """compute the conditional log_prob(x_{n_step} | x_{n_step-1}). When n_step = 0,
    log_prob(x_0) is calculated.
    """
    if n_step == 0:
      log_prob = np.diag(-0.5 * self.latent[0, :, :] @ np.linalg.inv(self.P) @ self.latent[0, :, :].T)
    else:
      diff = self.latent[n_step, :, :] - self.latent[n_step-1, :, :] @ self.A.T
      log_prob = np.diag(-0.5 * diff @ np.linalg.inv(self.B @ self.Q @ self.B.T) @ diff.T)
    return log_prob
      
  def log_alpha(self, n_step, proposal):
    """compute log of alpha_n(x_{1:n_step})
    Output:
      log_alpha_n: N
    """
    if n_step == 0:
      log_alpha_n = 1
    elif n_step > 0:
      log_alpha_n = self.log_prob_cond(n_step-1) - proposal.logcdf(self.latent[n_step, :, :])
    return log_alpha_n

  def log_weight(self, steps, proposal):
    """compute log of *unnormalized* weights w_n(x_n)
    Output:
      log_w_n: N
    """
    log_w_n = self.log_prob_cond(0) - proposal.logcdf(self.latent[0, :, :])
    for i in range(steps):
      log_w_n += self.log_alpha(i, proposal)

  def sis(self, proposal, steps=None):
    """sequential importance sampling"""
    steps = self.latent.shape[0]-1 if steps is None else steps

    posterior_samples = np.zeros((steps, self.N, self.dim_x))
    log_weights = np.zeros((steps, self.N))
    for i in range(steps):
      posterior_samples[i, :] = proposal.rvs(self.N)
      if i == 0:
        log_weights[i, :] = self.log_prob_cond(0) - proposal.logcdf(self.latent[0, :, :])
      else:
        log_weights[i, :] = log_weights[i-1, :] + self.log_alpha(i, proposal)
      
    return posterior_samples, log_weights
      