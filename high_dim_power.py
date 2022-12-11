
from src.ksd.ksd import KSD

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import tqdm, trange
from scipy.stats import norm


def ksd_gaussians(d, bandwidth, delta):
    lmda = bandwidth / 2.
    
    mean1 = tf.eye(d)[:, 0] * delta
    mean1_norm_sq = np.sum(mean1**2)
    
    res = (lmda / (lmda + 2))**(d/2) * mean1_norm_sq

    return res

def h1_var_gaussians(d, bandwidth, delta):
    lmda = bandwidth / 2.
    
    mean1 = tf.eye(d)[:, 0] * delta
    mean1_norm_sq = np.sum(mean1**2)

    # analytical form
    var_exp_up = (lmda / (lmda + 1))**(d/2) * (lmda / (lmda + 2))**(d/2) * (
        (lmda + 2) / (lmda + 1) * mean1_norm_sq + mean1_norm_sq**2
    )
    res = var_exp_up - ksd_gaussians(
        d=d, bandwidth=bandwidth, delta=delta
    )**2
    
    return res

def up_sq_gaussians(d, bandwidth, delta):
    lmda = bandwidth / 2.
    
    mean1 = tf.eye(d)[:, 0] * delta
    mean1_norm_sq = np.sum(mean1**2)

    res = (lmda / (lmda + 4))**(d/2) * (
        7 * d**2 / lmda + d + 6 * d / lmda * mean1_norm_sq + 2 * mean1_norm_sq + mean1_norm_sq**2
    )
    return res

def generate_target_proposal(dim, delta):
    # single gaussians
    
    mean1 = tf.eye(dim)[:, 0] * delta
    mean2 = tf.zeros(dim)

    target = tfd.MultivariateNormalDiag(mean1)
    proposal_off = tfd.MultivariateNormalDiag(mean2) 
    
    return target, proposal_off

def compute_population_quantities(
    dims: int,
    bandwidth_order: float,
    kernel_class,
    npop: int=4000,
    delta: float=2.,
):
    res = {
        "ksd": [],
        "cond_var": [],
        "full_var": [],
        "m3": [],
    }
    
    for d in tqdm(dims):
        # sample data
        target_dist, sample_dist = generate_target_proposal(d, delta)
        X = sample_dist.sample((npop,))
        
        # initialise KSD
        kernel = kernel_class(sigma_sq=2.*d**bandwidth_order)
        ksd = KSD(kernel=kernel, log_prob=target_dist.log_prob)
        
        # 1. KSD
        ksd_val = ksd(X, tf.identity(X)).numpy()
        # ksd_val = ksd_gaussians(d, bandwidth=kernel.sigma_sq, delta=delta)
        # 2. var_cond
        cond_var = ksd.h1_var(X=X, Y=tf.identity(X)).numpy()
        # cond_var = h1_var_gaussians(d, bandwidth=kernel.sigma_sq, delta=delta)
        # cond_var = ksd.u_p_abs_cond_central_moment(X=X, Y=tf.identity(X), k=2).numpy()
        # 3. var_full
        up_sq = ksd.u_p_moment(X, tf.identity(X), k=2).numpy()
        # up_sq = up_sq_gaussians(d, bandwidth=kernel.sigma_sq, delta=delta)
        full_var = up_sq - ksd_val**2
        # 4. m3
        m3 = ksd.u_p_abs_cond_central_moment(X, tf.identity(X), k=3).numpy()

        # store
        res["ksd"].append(ksd_val)
        res["cond_var"].append(cond_var)
        res["full_var"].append(full_var)
        res["m3"].append(m3)

    return res

def compute_ksd(
    ns,
    dims,
    nreps,
    kernel_class,
    bandwidth_order,
    delta: float=2.,
):
    """
    Compute KSD values for each dim d and sample size n.

    Args:
        t_ratio: t = (1 - n^{-\gamma}) * KSD * t_ratio.
    """
    res = {}
    iterator = tqdm(zip(dims, ns), total=len(dims))

    for d, n in iterator:
        target_dist, sample_dist = generate_target_proposal(d, delta)
        X = sample_dist.sample((nreps, n))
        
        # initialise KSD
        kernel = kernel_class(sigma_sq=2.*d**bandwidth_order)
        ksd = KSD(kernel=kernel, log_prob=target_dist.log_prob)
        
        # compute ksd
        ksd_vals = []
        for i in range(nreps):
            iterator.set_description(f"Repetition [{i+1}/{nreps}")

            ksd_val = ksd(X[i], tf.identity(X[i]))
            ksd_vals.append(ksd_val.numpy())
        
        res[d] = ksd_vals
        
        # ksd_vals =  ksd(X, tf.identity(X))
        # res[d] = ksd_vals.numpy()
        
    return res

def compute_analytical_power_bounds(
    n,
    dim,
    gamma,
    ksd,
    cond_var,
    full_var,
):
    upper_bd = (
        4 * cond_var / (n**(1 - 2*gamma) * ksd**2) + 
        4 * full_var / (n**(2 - 2*gamma) * ksd**2)
    )

    lower_bd = 1 - upper_bd
    
    return lower_bd, upper_bd

def compute_analytical_markov_bounds(
    n,
    t_lb,
    t_ub,
    ksd,
    m2,
    M2,
):
    upper_bd = (
        m2 / (n * (t_lb - ksd)**2) +
        M2 / (n * (n - 1) * ((t_lb - ksd)**2))
    )
    lower_bd = 1 - (
        m2 / (n * (t_ub - ksd)**2) +
        M2 / (n * (n - 1) * ((t_ub - ksd)**2))
    )
    
    return lower_bd, upper_bd

def compute_analytical_BE_bounds(
    n,
    t_lb,
    t_ub,
    ksd,
    m2,
    m3,
    M2,
):
    upper_bd = (
        norm.cdf(np.sqrt(n) * (ksd - t_ub) / (2 * m2**0.5)) +
        m2 / (n*(n - 1) * (t_ub - ksd)**2) +
        M2**0.5 * m2 / (n**(3/2) * (n - 1)**0.5 * (t_ub - ksd)**3) +
        m3 / (n**2 * (t_ub - ksd)**3)
    )

    lower_bd = 1 - (
        norm.cdf(np.sqrt(n) * (t_lb - ksd) / (2 * m2**0.5)) +
        m2 / (n*(n - 1) * (t_lb - ksd)**2) +
        M2**0.5 * m2 / (n**(3/2) * (n - 1)**0.5 * np.abs(t_lb - ksd)**3) +
        m3 / (n**2 * np.abs(  - ksd)**3)
    )

    return lower_bd, upper_bd

def power_experiment(
    ksd_res,
    res_analytical,
    ns,
    dims,
    eta,
    t_ratio,
):
    """
    Args:
        t_ratio: t = (1 - n^{-\gamma}) * KSD * t_ratio. Must
            lie in (0, 1)
        gamma: Must lie in (0, 1/2)
    """
    dims = np.array(dims)
    
    res = {
        "dim": dims,
        "n": ns,
        "probs_u": [],
        "probs_l": [],
        "u_bd": [],
        "l_bd": [],
        "t_u": [],
        "t_l": [],
    }
    
    for i, d in enumerate(tqdm(dims)):        
        n = ns[i]
        
        # compute analytical bounds
        l_bd, u_bd = compute_analytical_power_bounds(
            n,
            d,
            eta,
            ksd=res_analytical["ksd"][i],
            cond_var=res_analytical["cond_var"][i],
            full_var=res_analytical["full_var"][i],
        )
        
        res["l_bd"].append(l_bd)
        res["u_bd"].append(u_bd)
        
        # choose t
        t_l = (1 - n**(-eta)) * res_analytical["ksd"][i] * t_ratio
        t_u = (1 + n**(-eta)) * res_analytical["ksd"][i]

        res["t_l"].append(t_l)
        res["t_u"].append(t_u)

        # compute empirical probs
        ksd_vals = ksd_res[d]
        res["probs_l"].append(np.mean(ksd_vals >= t_l))
        res["probs_u"].append(np.mean(ksd_vals >= t_u))

    return res


def power_experiment_t(
    ksd_vals,
    res_analytical,
    n,
    ts_lb,
    ts_ub,
    bound: str="markov",
):
    """
    Plot power and bounds as a function of decision threshold t.
    """    
    res = {
        "bound": [bound] * len(ts_lb),
        "n": [n] * len(ts_lb),
        "probs_u": [],
        "probs_l": [],
        "u_bd": [],
        "l_bd": [],
        "t_u": [],
        "t_l": [],
    }
    
    for i, t_lb in enumerate(tqdm(ts_lb)):
        t_ub = ts_ub[i]
        
        if bound == "markov":
            l_bd, u_bd = compute_analytical_markov_bounds(
                n,
                t_lb,
                t_ub,
                ksd=res_analytical["ksd"],
                m2=res_analytical["cond_var"],
                M2=res_analytical["full_var"],
            )

        elif bound == "be":
            l_bd, u_bd = compute_analytical_BE_bounds(
                n,
                t_lb,
                t_ub,
                ksd=res_analytical["ksd"],
                m2=res_analytical["cond_var"],
                m3=res_analytical["m3"],
                M2=res_analytical["full_var"],
            )
        
        res["l_bd"].append(l_bd)
        res["u_bd"].append(u_bd)
        
        # save t
        res["t_l"].append(t_lb)
        res["t_u"].append(t_ub)

        # compute empirical probs
        res["probs_l"].append(np.mean(ksd_vals >= t_lb))
        res["probs_u"].append(np.mean(ksd_vals >= t_ub))

    return res