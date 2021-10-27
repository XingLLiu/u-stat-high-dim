import numpy as np

class KalmanFilter:
  def __init__(self, obs, prior_latent_mean, prior_latent_var, 
    A, B, C, D, R, Q):
    """
      obs: n x N x dy
    """
    self.prior_latent_mean = prior_latent_mean # dx
    self.prior_latent_var = prior_latent_var # dx x dx
    self.obs = obs
    self.n = obs.shape[0]

    self.A = A # dx x dx
    self.B = B # dx x dv
    self.C = C # dy x dx
    self.D = D # dy x dw
    self.Q = Q # dv x dv, symmetric
    self.v_var = D @ R @ D.T
    self.w_var = B @ Q @ B.T
    self.R = R # dw x dw, symmetric
    self.x1_0 = None # dx
    self.x1_1 = None # dx
    self.Sigma1_0 = None # dx x dx
    self.sigma1_1 = None # dx x dx

  def update_one_step(self, yn):
    """
      yn: N x dy
    """
    inv_mat = np.linalg.inv(self.C @ self.Sigma1_0 @ self.C.T + self.v_var) # dy x dy
    multiplier_mat = inv_mat @ self.C @ self.Sigma1_0.T # dy x dx
    self.x1_1 = self.x1_0 + (yn - self.x1_0 @ self.C.T) @ multiplier_mat # N x dx
    self.x1_0 = self.x1_1 @ self.A # N x dx
    self.Sigma1_1 = self.Sigma1_0 - self.Sigma1_0 @ self.C.T @ multiplier_mat # dx x dx
    self.Sigma1_0 = self.A @ self.Sigma1_1 @ self.A.T + self.w_var # dx x dx

  def fit(self, steps=None):
    """Run Kalman filter from initial state for 'steps' steps.
    Input:
      steps: how many steps to run Kalman filter algorithm. Must be <= self.obs.shape[0]
    Output:
      x1_1: E[x_n | y_{1:n}], N x dx
      Sigma1_1: error covariance matrix E[ (x_n - xn_n)(x_n - xn_n).T | y_{1:n} ], dx x dx
    """
    # reset state
    self.x1_0 = self.prior_latent_mean
    self.Sigma1_0 = self.prior_latent_var

    steps = self.n if steps is None else steps

    for i in range(steps):
      self.update_one_step(self.obs[i, :, :])

    return self.x1_1, self.Sigma1_1
