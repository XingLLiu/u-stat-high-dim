import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange

from src.ksd.find_modes import find_modes, pairwise_directions
import src.ksd.langevin as mcmc
from src.ksd.bootstrap import Bootstrap

class KSD:
  def __init__(
    self,
    kernel: tf.Module,
    target: tfp.distributions.Distribution=None,
    log_prob: callable=None
  ):
    """
    Inputs:
        target (tfp.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
        kernel (tf.nn.Module): [description]
        optimizer (tf.optim.Optimizer): [description]
    """
    if target is not None:
      self.p = target
      self.log_prob = target.log_prob
    else:
      self.p = None
      self.log_prob = log_prob
    self.k = kernel

  def u_p(self, X: tf.Tensor, Y: tf.Tensor, output_dim: int=1):
    """
    Inputs:
      X: n x dim
      Y: m x dim
    """
    # copy data for score computation
    X_cp = tf.identity(X)
    Y_cp = tf.identity(Y)

    ## calculate scores using autodiff
    if not hasattr(self.p, "grad_log"):
      with tf.GradientTape() as g:
        g.watch(X_cp)
        log_prob_X = self.log_prob(X_cp)
      score_X = g.gradient(log_prob_X, X_cp) # n x dim
      with tf.GradientTape() as g:
        g.watch(Y_cp)
        log_prob_Y = self.log_prob(Y_cp) # m x dim
      score_Y = g.gradient(log_prob_Y, Y_cp)
    else:
      score_X = self.p.grad_log(X_cp) # n x dim
      score_Y = self.p.grad_log(Y_cp) # m x dim
      assert score_X.shape == X_cp.shape

    # median heuristic #TODO using pre-specified bandwidth
    if self.k.med_heuristic:
      self.k.bandwidth(X, Y)

    # kernel
    K_XY = self.k(X, Y) # n x m

    # kernel grad
    grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
    grad_K_X = self.k.grad_first(X, Y) # n x m x dim
    
    # term 1
    term1_mat = tf.linalg.matmul(score_X, score_Y, transpose_b=True) * K_XY # n x m
    # term 2
    term2_mat = tf.expand_dims(score_X, -2) * grad_K_Y # n x m x dim
    term2_mat = tf.reduce_sum(term2_mat, axis=-1)
    # term3
    term3_mat = tf.expand_dims(score_Y, -3) * grad_K_X # n x m x dim
    term3_mat = tf.reduce_sum(term3_mat, axis=-1)
    # term4
    term4_mat = self.k.gradgrad(X, Y) # n x m

    u_p = term1_mat + term2_mat + term3_mat + term4_mat
    u_p_nodiag = tf.linalg.set_diag(u_p, tf.zeros(u_p.shape[:-1]))

    if output_dim == 1:
      ksd = tf.reduce_sum(
        u_p_nodiag,
        axis=[-1, -2]
      ) / (X.shape[-2] * (Y.shape[-2]-1))
      return ksd
    elif output_dim == 2:
      assert term1_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
      assert term2_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
      assert term3_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
      assert term4_mat.shape[-2:] == (X.shape[-2], Y.shape[-2]), term4_mat.shape
      return u_p_nodiag

  def u_p_moment(self, X: tf.Tensor, Y: tf.Tensor, k: int):
    u_p = self.u_p(X, Y, output_dim=2)
    u_p_sq = u_p**k
    u_p_sq = tf.math.reduce_sum(
      u_p_sq * (1 - tf.eye(u_p_sq.shape[-2]))
    ) / (X.shape[-2] * (X.shape[-2] - 1))
    return u_p_sq

  def __call__(self, X: tf.Tensor, Y: tf.Tensor, output_dim: int=1):
    """
    Inputs:
      X: n x dim
      Y: m x dim
    """
    return self.u_p(X, Y, output_dim)

  def h1_var(self, return_scaled_ksd: bool=False, **kwargs):
    """Compute the variance of the asymtotic Gaussian distribution under H_1
    Args:
      return_scaled_ksd: if True, return KSD / (\sigma_{H_1} + jitter), where 
        \sigma_{H_1}^2 is the asymptotic variance of the KSD estimate under H1
    """
    u_mat = self.__call__(output_dim=2, **kwargs) # n x n
    n = kwargs["X"].shape[-2]
    witness = tf.reduce_sum(u_mat, axis=1) # n
    term1 = tf.reduce_sum(witness**2) * 4 / n**3
    term2 = tf.reduce_sum(u_mat)**2 * 4 / n**4
    var = term1 - term2 + 1e-12
    if not return_scaled_ksd:
      return var
    else:
      ksd = tf.reduce_sum(u_mat) / n**2
      ksd_scaled = ksd / tf.math.sqrt(var)
      return var, ksd_scaled

  def test(self, x: tf.Tensor, num_boot: int):
    # get multinomial sample
    # Sampling can be slow. initialise separately for faster implementation
    n = x.shape[-2]
    bootstrap = Bootstrap(self, n)
    # (num_boot - 1) because the test statistic is also included 
    multinom_one_sample = bootstrap.multinom.sample((num_boot-1,)) # num_boot x ntest

    p_val = bootstrap.test_once(alpha=None, num_boot=num_boot, X=x, multinom_samples=multinom_one_sample)
    return bootstrap.ksd_hat, p_val


class PKSD(KSD):
  def __init__(self, 
    kernel: tf.Module, 
    pert_kernel: mcmc.RandomWalkMH,
    log_prob, callable=None,
    target: tfp.distributions.Distribution=None, 
    score: callable=None
  ):
      super().__init__(kernel, target, log_prob)
      self.pert_kernel = pert_kernel
      self.score = score

  def find_modes(self, init_points: tf.Tensor, threshold: float, **kwargs):
    # merge modes
    mode_list, inv_hess_list = find_modes(
      init_points, self.log_prob, threshold=threshold, grad_log=self.score, **kwargs)

    # find between-modes dir
    if len(mode_list) == 1:
        _, ind_pair_list = [mode_list[0]], [(0, 0)]
    else:
        _, ind_pair_list = pairwise_directions(mode_list, return_index=True)

    proposal_dict = mcmc.prepare_proposal_input_all(mode_list=mode_list, inv_hess_list=inv_hess_list)
    
    self.ind_pair_list = ind_pair_list
    self.proposal_dict = proposal_dict

  def test(self, 
    xtrain: tf.Tensor, 
    xtest: tf.Tensor, 
    T: int, 
    jump_ls: tf.Tensor,
    num_boot: int=1000, 
  ):
    """Finds the best jump scale using the training set, and use this jump scale to perform KSD 
    test using the perturbed samples.
    """
    if self.proposal_dict is None:
      raise ValueError("Must run find_modes before testing")

    # find best jump scale
    mh = self.pert_kernel(log_prob=self.log_prob)
    mh.run(steps=T, std=jump_ls, x_init=xtrain, ind_pair_list=self.ind_pair_list, **self.proposal_dict)

    # compute ksd
    scaled_ksd_vals = []
    for i in range(len(jump_ls)):
      # get samples after T steps
      x_t = mh.x[i, -1, :]

      if len(x_t.shape) == 1: 
          x_t = tf.expand_dims(x_t, -1)

      # compute ksd
      _, ksd_val = self.h1_var(X=x_t, Y=tf.identity(x_t), return_scaled_ksd=True)

      scaled_ksd_vals.append(ksd_val)

    # get best jump scale
    best_idx = tf.math.argmax(scaled_ksd_vals)
    best_jump = jump_ls[best_idx]
    
    # store samples corresponding to the best jump scale
    self.x = mh.x[best_idx]

    # run dynamic for T steps with test data and optimal params
    mh = self.pert_kernel(log_prob=self.log_prob)
    mh.run(steps=T, x_init=xtest, std=best_jump, ind_pair_list=self.ind_pair_list, **self.proposal_dict)

    # get perturbed samples
    x_t = mh.x[0, -1]
    if len(x_t.shape) == 1: 
        x_t = tf.expand_dims(x_t, -1)

    # get multinomial sample
    # Sampling can be slow. initialise separately for faster implementation
    ntest = xtest.shape[-2]
    bootstrap = Bootstrap(self, ntest)
    # (num_boot - 1) because the test statistic is also included 
    multinom_one_sample = bootstrap.multinom.sample((num_boot-1,)) # num_boot x ntest

    # compute p-value
    p_val = bootstrap.test_once(
      alpha=None,
      num_boot=num_boot,
      X=x_t,
      multinom_samples=multinom_one_sample,
    )
    
    return bootstrap.ksd_hat, p_val


class MPKSD(PKSD):
  def __init__(self, 
    kernel: tf.Module, 
    pert_kernel: mcmc.RandomWalkMH,
    log_prob, callable=None,
    target: tfp.distributions.Distribution=None, 
    score: callable=None
  ):
      super().__init__(kernel, target, log_prob)
      self.pert_kernel = pert_kernel
      self.score = score

  def __call__(
    self, 
    X: tf.Tensor, 
    Y: tf.Tensor, 
    Xp: tf.Tensor, 
    Yp: tf.Tensor, 
    output_dim: int=1,
  ):
    """
    Compute estimate of m-pKSD(Q, P) = pKSD(Q, P) + KSD(Q, P).
    
    Args:
      X, Y: Unperturbed data.
      Xp, Yp: Perturbed data.
    """
    ksd_u_p = self.u_p(X, Y, output_dim)
    pksd_u_p = self.u_p(Xp, Yp, output_dim)

    return ksd_u_p + pksd_u_p

  def test(self, 
    xtrain: tf.Tensor, 
    xtest: tf.Tensor, 
    T: int, 
    jump_ls: tf.Tensor,
    num_boot: int=1000, 
  ):
    """Finds the best jump scale using the training set, and use this jump scale to perform KSD 
    test using the perturbed samples.
    """
    if self.proposal_dict is None:
      raise ValueError("Must run find_modes before testing")

    # find best jump scale
    mh = self.pert_kernel(log_prob=self.log_prob)
    mh.run(steps=T, std=jump_ls, x_init=xtrain, ind_pair_list=self.ind_pair_list, **self.proposal_dict)

    # compute ksd
    scaled_ksd_vals = []
    for i in range(len(jump_ls)):
      # get samples after T steps
      x_0 = mh.x[i, 0, :]
      x_t = mh.x[i, -1, :]
      assert tf.rank(x_t.shape) == 1

      if len(x_t.shape) == 1: 
          x_t = tf.expand_dims(x_t, -1)

      # compute ksd
      _, ksd_val = self.h1_var(
        X=x_0,
        Y=tf.identity(x_0),
        Xp=x_t,
        Yp=tf.identity(x_t),
        return_scaled_ksd=True,
      )

      scaled_ksd_vals.append(ksd_val)

    # get best jump scale
    best_idx = tf.math.argmax(scaled_ksd_vals)
    best_jump = jump_ls[best_idx]
    self.best_jump_scale = best_jump
    
    # store samples corresponding to the best jump scale
    self.x = mh.x[best_idx]

    # run dynamic for T steps with test data and optimal params
    mh = self.pert_kernel(log_prob=self.log_prob)
    mh.run(steps=T, x_init=xtest, std=best_jump, ind_pair_list=self.ind_pair_list, **self.proposal_dict)

    # get perturbed samples
    x_0 = mh.x[0, 0]
    x_t = mh.x[0, -1]
    assert tf.rank(x_t.shape) == 1
    if len(x_t.shape) == 1: 
        x_t = tf.expand_dims(x_t, -1)

    # get multinomial sample
    # Sampling can be slow. initialise separately for faster implementation
    ntest = xtest.shape[-2]
    bootstrap = Bootstrap(self, ntest)
    # (num_boot - 1) because the test statistic is also included 
    multinom_one_sample = bootstrap.multinom.sample((num_boot-1,)) # num_boot x ntest

    # compute p-value
    p_val = bootstrap.test_once(
      alpha=None,
      num_boot=num_boot,
      X=x_t,
      multinom_samples=multinom_one_sample,
      X_unperturbed=x_0,
      statistic="m_pksd",
    )
    
    return bootstrap.ksd_hat, p_val


class SPKSD(PKSD):
  def __init__(self, 
    kernel: tf.Module, 
    pert_kernel: mcmc.RandomWalkMH,
    log_prob, callable=None,
    target: tfp.distributions.Distribution=None, 
    score: callable=None
  ):
      super().__init__(kernel, target, log_prob)
      self.pert_kernel = pert_kernel
      self.score = score

  def __call__(
    self, 
    X: tf.Tensor, 
    Y: tf.Tensor,
    output_dim: int=1,
  ):
    """
    Compute estimate of spKSD(Q, P) = \sum_j KSD(\mathcal{K}_j Q, P).
    
    Args:
      X, Y: Array of perturbed data.
    """
    self.pksd_vals = []
    pksd_sum = 0.
    for i in range(X.shape[-3]):
      self.pksd_vals.append(self.u_p(X[i], Y[i], output_dim))
      pksd_sum += self.pksd_vals[i]

    return pksd_sum

  def test(self, 
    x: tf.Tensor,
    T: int, 
    jump_ls: tf.Tensor,
    num_boot: int=1000, 
  ):
    """Finds the best jump scale using the training set, and use this jump scale to perform KSD 
    test using the perturbed samples.
    """
    if self.proposal_dict is None:
      raise ValueError("Must run find_modes before testing")

    # find best jump scale
    mh = self.pert_kernel(log_prob=self.log_prob)
    mh.run(steps=T, std=jump_ls, x_init=x, ind_pair_list=self.ind_pair_list, **self.proposal_dict)

    # compute pksd
    x_t = mh.x[:, -1, :]
    x_0 = tf.expand_dims(x, axis=0)
    x_t = tf.concat([x_0, x_t], axis=0) # J x n x dim

    # get multinomial sample
    # Sampling can be slow. initialise separately for faster implementation
    bootstrap = Bootstrap(self, x_t.shape[-2])
    # (num_boot - 1) because the test statistic is also included 
    multinom_one_sample = bootstrap.multinom.sample((num_boot-1,)) # num_boot x ntest

    # compute p-value
    p_val = bootstrap.test_once(
      alpha=None,
      num_boot=num_boot,
      X=x_t,
      multinom_samples=multinom_one_sample,
      statistic="pksd",
    )
    
    return bootstrap.ksd_hat, p_val