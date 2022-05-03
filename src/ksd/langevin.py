import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import trange

def prepare_proposal_input(mode1: tf.Tensor, mode2: tf.Tensor, hess1_inv: tf.Tensor, hess2_inv: tf.Tensor):
    """Given two modes and the local inverse Hessians, compute the
    quantities needed to construct the proposals.
    
    Args:
        mode1, mode2: 1 x dim
        hess1_inv, hess2_inv: dim x dim, inverse Hessians at the modes
    """
    # if Hessian estimates were not stable, set to I_d
    if tf.linalg.det(hess1_inv) <= 0:
      hess1_inv = tf.eye(hess1_inv.shape[0])
    if tf.linalg.det(hess2_inv) <= 0:
      hess2_inv = tf.eye(hess2_inv.shape[0])

    hess1_inv_sqrt = tf.linalg.sqrtm(hess1_inv)
    hess2_inv_sqrt = tf.linalg.sqrtm(hess2_inv)
    hess1_sqrt = tf.linalg.inv(hess1_inv_sqrt)
    hess2_sqrt = tf.linalg.inv(hess2_inv_sqrt)

    hess1_sqrt_det = tf.linalg.det(hess1_sqrt)
    hess1_inv_sqrt_det = tf.linalg.det(hess1_inv_sqrt)
    hess2_sqrt_det = tf.linalg.det(hess2_sqrt)
    hess2_inv_sqrt_det = tf.linalg.det(hess2_inv_sqrt)
    hess_dict = {
        "mode1": mode1,
        "hess1_sqrt": hess1_sqrt, "hess1_inv_sqrt": hess1_inv_sqrt,
        "hess1_sqrt_det": hess1_sqrt_det, "hess1_inv_sqrt_det": hess1_inv_sqrt_det,
        "mode2": mode2,
        "hess2_sqrt": hess2_sqrt, "hess2_inv_sqrt": hess2_inv_sqrt,
        "hess2_sqrt_det": hess2_sqrt_det, "hess2_inv_sqrt_det": hess2_inv_sqrt_det,
    }
    return hess_dict

def prepare_proposal_input_all(mode_list: list, inv_hess_list: list):
    """Given a list of modes and the local inverse Hessians, compute the
    quantities needed to construct the proposals.
    
    Args:
        mode_list: list of modes of shape (dim,)
        inv_hess_list: list of inverse Hessians at the modes, of shape dim x dim
    """
    modes = tf.stack(mode_list) # nmodes x dim

    # if Hessian estimates were not stable, set to I_d
    for i, inv_hess in enumerate(inv_hess_list):
      if tf.linalg.det(inv_hess) <= 0:
        inv_hess_list[i] = tf.eye(inv_hess.shape[0])

    inv_hess_sqrt_list = [tf.linalg.sqrtm(x) for x in inv_hess_list]
    inv_hess_sqrt = tf.stack(inv_hess_sqrt_list) # nmodes x dim x dim

    hess_sqrt_list = [tf.linalg.inv(x) for x in inv_hess_sqrt_list]
    hess_sqrt = tf.stack(hess_sqrt_list) # nmodes x dim x dim
  
    hess_sqrt_det_list = [tf.linalg.det(x) for x in hess_sqrt_list]
    hess_sqrt_det = tf.stack(hess_sqrt_det_list) # nmodes

    inv_hess_sqrt_det_list = [tf.linalg.det(x) for x in inv_hess_sqrt_list]
    inv_hess_sqrt_det = tf.stack(inv_hess_sqrt_det_list) # nmodes
    
    modes_dict = {
        "modes": modes,
        "hess_sqrt": hess_sqrt,
        "inv_hess_sqrt": inv_hess_sqrt,
        "hess_sqrt_det": hess_sqrt_det,
        "inv_hess_sqrt_det": inv_hess_sqrt_det,
    }
    return modes_dict

class MCMC:
  def __init__(self, log_prob: callable) -> None:
    self.log_prob = log_prob
    self.x = None
    self.noise_dist = tfd.Normal(0., 1.)

  def log_transition_kernel(self, xp: tf.Tensor, x: tf.Tensor):
    '''Compute log k(x'| x), where k is the transition kernel 

    Args:
      x_proposed: n x dim. Proposed samples
      x_current: n x dim. Samples at time t-1
      score: n x dim. Score evaluated at x_current

    Output:
      log_kernel: n. k(x', x)
    '''
    raise NotImplementedError(
      "Sampler class '{:s}' does not provide a 'transition kernel'".format(self._get_class_name())
    )

  def compute_accept_prob(self, x_proposed: tf.Tensor, x_current: tf.Tensor,
    log_det_jacobian: tf.Tensor, **kwargs):
    '''Compute log acceptance prob in the Metropolis step,
    log( \min(1, k(x, x') * p(x') / (k(x', x) * p(x))) )
    '''
    log_numerator = self.log_prob(x_proposed) + self.log_transition_kernel(
      xp=x_current, x=x_proposed, **kwargs
    ) + log_det_jacobian # n
    log_denominator = self.log_prob(x_current) + self.log_transition_kernel(
      xp=x_proposed, x=x_current, **kwargs
    ) # n
    # print(x_proposed[:2])
    # print(self.log_prob(x_proposed), self.log_prob(x_current))
    log_prob = tf.math.minimum(0., log_numerator - log_denominator)
    return tf.math.exp(log_prob)

  def metropolis(self, x_proposed: tf.Tensor, x_current: tf.Tensor, accept_prob: tf.Tensor):
    '''Metropolis move according to accept_prob

    Args:
      accept_prob: n. Tensor of alpha_i, i = 1, ..., n.

    Output:
      x_next: n x dim. new particles
    '''
    unif = tf.random.uniform(shape=accept_prob.shape) # n
    cond = tf.expand_dims(unif < accept_prob, axis=-1) # n x 1
    x_next = tf.where(cond, x_proposed, x_current) # n x dim
    if_accept = tf.reshape(tf.cast(cond, dtype=tf.float32), (-1,)) # n

    assert x_next.shape == x_proposed.shape
    assert tf.experimental.numpy.all(tf.where(cond, x_next == x_proposed, x_next == x_current))
    return x_next, if_accept


class RandomWalkMH(MCMC):
  def __init__(self, log_prob: callable) -> None:
    self.log_prob = log_prob
    self.x = None

  def run(self, steps: int, x_init: tf.Tensor, verbose: bool=False, **kwargs):
    n, dim = x_init.shape
    self.x = tf.Variable(-1 * tf.ones((steps, n, dim))) # nsteps x n x dim
    self.x[0, :, :].assign(x_init)

    if "ind_pair_list" in kwargs: # use all modes for proposal
      npairs = len(kwargs["ind_pair_list"])
      ind_prob = [1/npairs] * npairs
      self.ind_pair_sample = tfp.distributions.Categorical(ind_prob).sample((steps-1, n)) # steps-1 x n
      self.ind_pairs = tf.constant(kwargs["ind_pair_list"]) # (nmodes * (nmodes - 1) / 2) x 2

    self.accept_prob = tf.Variable(tf.zeros((steps-1, n))) # (steps-1) x n
    self.if_accept = tf.Variable(tf.zeros((steps-1, n))) # (steps-1) x n

    iterator = trange(steps-1) if verbose else range(steps-1)

    for t in iterator: 
      self.t = t
      
      # propose MH update
      x_current = self.x[t, :, :]
      xp_next, log_det_jacobian = self.proposal(x_current=x_current, **kwargs) # n x dim #! delete

      # compute acceptance prob
      accept_prob = self.compute_accept_prob(
        x_proposed=xp_next,
        x_current=x_current,
        log_det_jacobian=log_det_jacobian,
        **kwargs
      ) # n
      self.accept_prob[t, :].assign(accept_prob)
      
      # move
      x_next, if_accept = self.metropolis(x_proposed=xp_next, x_current=x_current, accept_prob=accept_prob) # n x dim, n x 1
      # x_next = xp_next #! delete
      self.if_accept[t, :].assign(if_accept)

      # store next samples
      self.x[t+1, :, :].assign(x_next)

  def log_transition_kernel(self, xp: tf.Tensor, x: tf.Tensor, std: float, **kwargs):
    '''Compute log k(x'| x), where k is the transition kernel 
    k(x' | x) \propto N(x' | x, std**2)
    
    Args:
      x_proposed: n x dim. Proposed samples
      x_current: n x dim. Samples at time t-1
      score: n x dim. Score evaluated at x_current

    Output:
      log_kernel: n. k(x', x)
    '''
    log_prob = tf.math.log(0.5) * tf.ones(xp.shape[0])

    return log_prob

  def proposal(self, x_current, **kwargs):
    """Default proposal:

    x_current: n x dim
    """
    std = kwargs["std"]

    if "ind_pair_list" in kwargs: # use all modes for proposal #TODO new proposal
      ind_pair_ind = self.ind_pair_sample[self.t, :] # n x 1
      ind_pair = tf.gather(self.ind_pairs, ind_pair_ind) # n x 2
      
      # inverse hessian = covariance matrix
      mode1 = tf.gather(kwargs["modes"], ind_pair[:, 0]) # n x dim
      mode2 = tf.gather(kwargs["modes"], ind_pair[:, 1]) # n x dim
      inv_root_cov_1 = tf.gather(kwargs["hess_sqrt"], ind_pair[:, 0]) # n x dim x dim
      root_cov_2 = tf.gather(kwargs["inv_hess_sqrt"], ind_pair[:, 1]) # n x dim x dim
      inv_root_cov_1_det = tf.gather(kwargs["hess_sqrt_det"], ind_pair[:, 0]) # n
      root_cov_2_det = tf.gather(kwargs["inv_hess_sqrt_det"], ind_pair[:, 1]) # n

      # construct proposals
      x_current1 = tf.expand_dims(x_current - std * mode1, axis=1) # n x 1 x dim
      xp_next = tf.squeeze(x_current1 @ inv_root_cov_1 @ root_cov_2, axis=1) + std * mode2 # n x dim
      
      n = xp_next.shape[0]
      det_jacobian = inv_root_cov_1_det * root_cov_2_det # n
      log_det_jacobian = tf.math.log(det_jacobian) # n

    return xp_next, log_det_jacobian

class RandomWalkBarker(RandomWalkMH):
  def __init__(self, log_prob: callable) -> None:
    super().__init__(log_prob)

  def compute_accept_prob(self, x_proposed: tf.Tensor, x_current: tf.Tensor,
    log_det_jacobian: tf.Tensor, **kwargs):
    '''Compute log acceptance prob using the Barker's rule, 
    log( k(x, x') * p(x') / (k(x', x) * p(x) + k(x, x') * p(x')) )
    '''
    term_xp = tf.exp(
      self.log_prob(x_proposed) + self.log_transition_kernel(
        xp=x_current, x=x_proposed, **kwargs
      ) + log_det_jacobian
    ) # n
    term_x = tf.exp(
      self.log_prob(x_current) + self.log_transition_kernel(
        xp=x_proposed, x=x_current, **kwargs
      )
    ) # n
    prob = term_xp / (term_xp + term_x) # n
    return prob