import autograd.numpy as anp
import kgof.density as kgof_density
from scipy.special import logsumexp


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


def multivariate_t_logprob(x, loc, Sigma_inv, df, dim):
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
    diff = x - loc
    prod = anp.matmul(diff, Sigma_inv) # n x d
    prod = anp.einsum("ij,ij->i", prod, diff) # n
    prod /= df
    log_den = -0.5 * (df + dim) * anp.log(1 + prod)
    return log_den


def banana_logprob(x, loc, b, df, dim, scale: float=10.):
    '''
    '''
    id_mat = anp.eye(dim)
    scale_mat = anp.concatenate([id_mat[:, :1] * scale, id_mat[:, 1:]], axis=-1)
    Sigma = anp.matmul(scale_mat, anp.transpose(scale_mat))
    Sigma_inv = anp.linalg.inv(Sigma)

    x_0 = x[..., 0:1]
    x[..., 1:2] = x[..., 1:2] - b * x_0**2 + 100 * b
    # print("x", x[:10])
    # print("params", loc, Sigma_inv, df, dim)
    log_den = multivariate_t_logprob(x, loc, Sigma_inv, df, dim)
    # print("log_den", log_den[:10])
    return log_den


def create_mixture_t_banana_logprob(dim, ratio, loc, nbanana, std, b):
  '''
  '''
  nmodes = len(ratio)
  assert nmodes >= nbanana, f"number of mixtures {nmodes} must be >= {nbanana}"
  
  t_scale = anp.sqrt(std * anp.sqrt(float(dim)) * anp.eye(dim))
  Sigma = t_scale @ anp.transpose(t_scale)
  Sigma_inv = anp.linalg.inv(Sigma)

  ratio = anp.array(ratio).reshape((-1, 1))
  # log_ratio = [anp.log(r) for r in ratio]
  # print("ratio", ratio, "log_ratio", log_ratio)

  def log_prob_fn(x):
    b_log_probs = [
      banana_logprob(
        x,
        loc=loc[i],
        b=b,
        df=7,
        dim=dim,
      ) for i in range(nbanana)
    ]
    # b_log_probs = [log_ratio[i] + b_log_probs[i] for i in range(nbanana)]

    t_log_probs = [
      multivariate_t_logprob(
        x, 
        loc=loc[nbanana+i], 
        Sigma_inv=Sigma_inv, 
        df=7, 
        dim=dim
      ) for i in range(nmodes-nbanana)
    ]
    # t_log_probs = [log_ratio[nbanana+i] + t_log_probs[i] for i in range(nmodes-nbanana)]

    log_probs = b_log_probs + t_log_probs

    # print("b", [p[:10] for p in b_log_probs])
    # print("t", [p[:10] for p in t_log_probs])
    log_prob = logsumexp(anp.stack(log_probs, axis=0), axis=0, b=ratio)
    return log_prob

  return log_prob_fn


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
