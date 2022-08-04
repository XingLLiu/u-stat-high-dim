import autograd.numpy as anp
import kgof.density as kgof_density


def assert_equal_log_prob(dist, log_prob, log_prob_np):
  """Check if log_prob == dist.log_prob + const."""
  x = dist.sample(10)

  res = anp.allclose(log_prob(x), log_prob_np(anp.array(x)), atol=1e-5)
  assert res, "log_prob and log_prob_np yield different values"

# end checker for log_prob_np implementation

def create_mixture_gaussian_kdim_logprobb(dim, k, delta, ratio=0.5, shift=0.):
    """
    Evaluate the log density at the points (rows) in X 
    of the standard isotropic Gaussian.
    Note that the density is NOT normalized. 
    
    X: n x d nd-array
    return a length-n array
    """
    a = [1. if x < k else 0. for x in range(dim)]
    a = anp.array(a)
    multiplier = delta / anp.sqrt(float(k))
    mean1 = anp.zeros(dim) + shift
    mean2 = multiplier * a + shift

    log_ratio1 = anp.log(ratio)
    log_ratio2 = anp.log(1-ratio)
    
    variance = 1

    def log_prob_fn(X):
      exp1 = -0.5 * anp.sum((X-mean1)**2, axis=-1) / variance + log_ratio1
      exp2 = -0.5 * anp.sum((X-mean2)**2, axis=-1) / variance + log_ratio2
      unden = anp.logaddexp(exp1, exp2) # n
      return unden

    return log_prob_fn


# def multivariate_t_prob(x, Sigma_inv, df, dim):
#     '''
#     Multivariate t-student density:
#     output:
#         the density of the given element
#     input:
#         x = parameter (d dimensional numpy array or scalar)
#         mu = mean (d dimensional numpy array or scalar)
#         Sigma = scale matrix (dxd numpy array)
#         df = degrees of freedom
#         d: dimension
#     '''
#     diff = x
#     prod = anp.matmul(diff, Sigma_inv) # n x d
#     prod = anp.einsum("ij,ij->i", prod, diff) # n
#     prod /= df
#     den = anp.power(1 + prod, -(df + dim) * 0.5)
#     return den


# def banana_prob(x, mu, scale, b, df, dim):
#     '''
#     '''
#     id_mat = anp.eye(dim)
#     scale_mat = anp.concatenate([id_mat[:, :1] * scale, id_mat[:, 1:]], axis=-1)
#     Sigma_inv_banana = anp.linalg.inv(scale_mat**2)

#     x_0 = x[..., 0:1]
#     x[..., 1:2] = x[..., 1:2] - b * x_0**2 + 100 * b
    
#     den = multivariate_t_prob(x, Sigma_inv_banana, df, dim)
#     return den


def multivariate_t_prob(x, df, dim):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    '''
    prod = anp.einsum("ij,ij->i", x, x) # n
    prod /= df
    den = anp.power(1 + prod, -(df + dim) * 0.5)
    return den


def banana_prob(x, mu, scale, b, df, dim):
    '''
    '''
    id_mat = anp.eye(dim)
    scale_mat = anp.concatenate([id_mat[:, :1] * scale, id_mat[:, 1:]], axis=-1)
    scale_inv_banana = anp.linalg.inv(scale_mat)

    x_0 = x[..., 0:1]
    x[..., 1:2] = x[..., 1:2] - b * x_0**2 + 100 * b

    y = anp.matmul(x - mu, scale_inv_banana)
    
    den = multivariate_t_prob(y, df, dim)
    return den


def create_rbm(
  B_scale: anp.array=8.,
  c: anp.array=0.,
  dx: int=50,
  dh: int=40,
  ):
  """
  Generate data for the Gaussian-Bernoulli Restricted Boltzmann Machine (RBM) experiment.
  The entries of the matrix B are perturbed.
  This experiment was first proposed by Liu et al., 2016 (Section 6)
  Args:
    m: number of samples
    c: (dh,) either tf.Tensor or set to tf.zeros((dh,)) by default
    sigma: standard deviation of Gaussian noise
    dx: dimension of observed output variable
    dh: dimension of binary latent variable
    burnin_number: number of burn-in iterations for Gibbs sampler
  """
  # Model p
  B = anp.eye(anp.max((dx, dh)))[:dx, :dh] * B_scale
  b = anp.zeros(dx)
  c = c if isinstance(c, anp.ndarray) else anp.zeros(dh)

  dist = kgof_density.GaussBernRBM(B, b, c)
  dist.log_prob = dist.log_den

  return dist.log_prob
