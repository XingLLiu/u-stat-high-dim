
from src.ksd.ksd import KSD
from src.mmd.mmd import MMD
from src.ksd.bootstrap import Bootstrap
import high_dim.analytical as hd_ana

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import tqdm, trange
import scipy.stats as spy_stats


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

# class MultivariateNormal(tfd.MultivariateNormalDiag):
#     """
#     Multivariate Gaussian with diagonal covariance matrix. If used
#     for KSD, the covariance matrix must be identity.
#     """
#     def __init__(self, mean, **kwargs):
#         super().__init__(mean, **kwargs)
#         self.mean = mean
        
#     def grad_log(self, x):
#         return - (x - self.mean)


class MultivariateNormal(tfd.Distribution):
    """
    Multivariate Gaussian with diagonal covariance matrix. If used
    for KSD, the covariance matrix must be identity.
    """
    def __init__(self, mean, **kwargs):
        self.mean = mean
        self.dist = tfd.MultivariateNormalDiag(mean, **kwargs)
        
    def log_prob(self, x):
        return self.dist.log_prob(x)

    def sample(self, size):
        return self.dist.sample(size)

    def grad_log(self, x):
        return - (x - self.mean)

def generate_target_proposal(dim, delta):
    # single gaussians
    
    mean1 = tf.eye(dim)[:, 0] * delta
    mean2 = tf.zeros(dim)

    # target = tfd.MultivariateNormalDiag(mean1)
    # proposal_off = tfd.MultivariateNormalDiag(mean2) 

    target = MultivariateNormal(mean1)
    proposal_off = MultivariateNormal(mean2) 
    
    return target, proposal_off

# # TODO non-id cov matrix
def generate_target_proposal_general_cov(dim, delta):
    """
        Mean for the alternative is \mu_1 = (0, delta, 0, ..., 0)^T

        Cov matrix \Sigma is diagonal with \Sigma_{ii} = 0.5*d
        and \Sigma_{jj} = 0.5 for j > 1.
    """    
    mean1 = tf.eye(dim)[:, 1] * delta
    mean2 = tf.zeros(dim)

    diag_mat = np.ones((dim), dtype=np.float32) * np.sqrt(0.5)
    diag_mat[0] = tf.math.sqrt(0.5 * (dim+1))
    diag_mat = tf.constant(diag_mat)

    target = tfd.MultivariateNormalDiag(mean1, scale_diag=diag_mat)
    proposal_off = tfd.MultivariateNormalDiag(mean2, scale_diag=diag_mat)

    return target, proposal_off

def generate_target_proposal_t(dim, delta, df=5):
    # single gaussians
    
    mean1 = tf.eye(dim)[:, 0] * delta
    mean2 = tf.zeros(dim)

    target = tfd.MultivariateStudentTLinearOperator(
        df=df,
        loc=mean1,
        scale=tf.linalg.LinearOperatorLowerTriangular(tf.eye(dim)),
    )
    proposal_off = tfd.MultivariateStudentTLinearOperator(
        df=df,
        loc=mean2,
        scale=tf.linalg.LinearOperatorLowerTriangular(tf.eye(dim)),
    )

    return target, proposal_off

def compute_population_quantities(
    dims: int,
    bandwidth_order: float,
    kernel_class,
    npop: int=4000,
    delta: float=2.,
    statistic: str="ksd",
    bandwidth_scale: float=2.,
):
    res = {
        "expectation": [],
        "cond_var": [],
        "full_var": [],
        "m3": [],
        "bandwidth": [],
    }
    
    for d in tqdm(dims):
        # initialise kernel with median heuristic
        bandwidth = bandwidth_scale*d**bandwidth_order
        kernel = kernel_class(sigma_sq=bandwidth)
        
        # sample data
        # TODO
        if kernel_class.__name__ == "RBF":
            target_dist, sample_dist = generate_target_proposal(d, delta)
            # target_dist, sample_dist = generate_target_proposal_t(d, delta)
        elif kernel_class.__name__ == "Linear":
            target_dist, sample_dist = generate_target_proposal_general_cov(d, delta)

        if statistic == "ksd":
            X = sample_dist.sample((npop,))
            Y = tf.identity(X)
            # initialise KSD
            # metric_fn = KSD(kernel=kernel, log_prob=target_dist.log_prob)
            metric_fn = KSD(kernel=kernel, target=target_dist)

        elif statistic == "mmd":
            X = sample_dist.sample((npop,))
            Y = target_dist.sample((npop,))
            # initialise MMD
            metric_fn = MMD(kernel=kernel)        
        
        # 1. expectation
        expectation_val = metric_fn(X, Y).numpy()
        # ksd_val = ksd_gaussians(d, bandwidth=kernel.sigma_sq, delta=delta)

        # 2. var_cond
        cond_var = metric_fn.abs_cond_central_moment(X=X, Y=Y, k=2).numpy()
        # cond_var = h1_var_gaussians(d, bandwidth=kernel.sigma_sq, delta=delta)

        # 3. var_full
        # up_sq = up_sq_gaussians(d, bandwidth=kernel.sigma_sq, delta=delta)
        # up_sq = metric_fn.u_p_moment(X, Y, k=2).numpy()
        # full_var = up_sq - expectation_val**2
        full_var = metric_fn.abs_full_central_moment(X=X, Y=Y, k=2).numpy()

        # 4. m3
        m3 = metric_fn.abs_cond_central_moment(X, Y, k=3).numpy()

        # store
        res["expectation"].append(expectation_val)
        res["cond_var"].append(cond_var)
        res["full_var"].append(full_var)
        res["m3"].append(m3)
        res["bandwidth"].append(bandwidth)

    return res

def compute_population_quantities_exact(
    dims: int,
    bandwidth_order: float,
    kernel_class,
    delta: float=2.,
    statistic: str="ksd",
    bandwidth_scale: float=2.,
):
    res = {
        "expectation": [],
        "cond_var": [],
        "full_var": [],
        "bandwidth": [],
    }
    
    for d in tqdm(dims):
        # initialise kernel with median heuristic
        bandwidth = bandwidth_scale*d**bandwidth_order
        kernel = kernel_class(sigma_sq=bandwidth)
        
        # sample data
        # TODO
        if kernel_class.__name__ == "RBF":
            target_dist, sample_dist = generate_target_proposal(d, delta)
            # target_dist, sample_dist = generate_target_proposal_t(d, delta)
        elif kernel_class.__name__ == "Linear":
            target_dist, sample_dist = generate_target_proposal_general_cov(d, delta)

        if statistic == "ksd":
            stat_ana = hd_ana.KSDAnalytical(
                dim=d,
                mu_norm=delta,
                bandwidth_power=bandwidth_order,
                bandwidth_scale=bandwidth_scale,
            )

        elif statistic == "mmd":
            if kernel_class.__name__ == "RBF":
                stat_ana = hd_ana.MMDAnalytical(
                    dim=d,
                    mu_norm=delta,
                    bandwidth_power=bandwidth_order,
                    bandwidth_scale=bandwidth_scale,
                )

            elif kernel_class.__name__ == "Linear":
                # TODO change with non-standard Gaussian model
                mu = np.eye(d)[:, 1] * delta

                sigma_mat = np.eye(d, dtype=np.float32) * 0.5
                sigma_mat[0, 0] = 0.5 * d
                
                stat_ana =  hd_ana.MMDLinearAnalytical(
                    dim=d,
                    mu=mu,
                    Sigma=sigma_mat,
                )
            
        # store
        res["expectation"].append(stat_ana.mean())
        res["cond_var"].append(stat_ana.cond_var())
        res["full_var"].append(stat_ana.full_var())
        res["bandwidth"].append(bandwidth)

    return res

def compute_statistic(
    ns,
    dims,
    nreps,
    kernel_class,
    bandwidth_order,
    delta: float=2.,
    statistic: str="ksd",
    bandwidth_scale: float=2.,
):
    """
    Compute, for nreps repetitions, the value of statistic for 
    each dim d and sample size n.

    Args:
        t_ratio: t = (1 - n^{-\gamma}) * KSD * t_ratio.
    """
    res = {}
    iterator = tqdm(zip(dims, ns), total=len(dims))

    for d, n in iterator:
        # initialise kernel with median heuristic
        bandwidth = bandwidth_scale*d**bandwidth_order
        kernel = kernel_class(sigma_sq=bandwidth)

        # sample data
        # TODO
        if kernel_class.__name__ == "RBF":
            target_dist, sample_dist = generate_target_proposal(d, delta)
            # target_dist, sample_dist = generate_target_proposal_t(d, delta)
        elif kernel_class.__name__ == "Linear":
            target_dist, sample_dist = generate_target_proposal_general_cov(d, delta)

        if statistic == "ksd":
            X = sample_dist.sample((nreps, n))
            Y = tf.identity(X)
            
            # initialise KSD
            # metric_fn = KSD(kernel=kernel, log_prob=target_dist.log_prob)
            metric_fn = KSD(kernel=kernel, target=target_dist)
        
        elif statistic == "mmd":
            X = sample_dist.sample((nreps, n))
            Y = target_dist.sample((nreps, n))
        
            # initialise MMD
            metric_fn = MMD(kernel=kernel)
                
        # compute statistic
        val_list = []
        for i in range(nreps):
            iterator.set_description(f"Repetition [{i+1}/{nreps}")

            val = metric_fn(X[i], Y[i])
            val_list.append(val.numpy())
        
        res[d] = val_list
                
    return res

def compute_statistic_rep(
    ns,
    dims,
    nreps,
    kernel_class,
    bandwidth_order,
    nexperiments,
    delta: float=2.,
    statistic: str="ksd",
    print_once_every: int=1,
    bandwidth_scale: float=2.,
):
    """
    Repeat compute_statistic for nexperiments times.
    """
    res_list = []
    for i in range(nexperiments):
        if i % print_once_every == 0:
            print(f"[{i+1} / {nexperiments}]")

        res = compute_statistic(
            ns,
            dims,
            nreps,
            kernel_class,
            bandwidth_order,
            delta=delta,
            statistic=statistic,
            bandwidth_scale=bandwidth_scale,
        )
        res_list.append(res)
    
    return res_list

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
        spy_stats.norm.cdf(np.sqrt(n) * (ksd - t_ub) / (2 * m2**0.5)) +
        m2 / (n*(n - 1) * (t_ub - ksd)**2) +
        M2**0.5 * m2 / (n**(3/2) * (n - 1)**0.5 * (t_ub - ksd)**3) +
        m3 / (n**2 * (t_ub - ksd)**3)
    )

    lower_bd = 1 - (
        spy_stats.norm.cdf(np.sqrt(n) * (t_lb - ksd) / (2 * m2**0.5)) +
        m2 / (n*(n - 1) * (t_lb - ksd)**2) +
        M2**0.5 * m2 / (n**(3/2) * (n - 1)**0.5 * np.abs(t_lb - ksd)**3) +
        m3 / (n**2 * np.abs(  - ksd)**3)
    )

    return lower_bd, upper_bd

def compute_analytical_new_full_bounds(
    n,
    t_lb,
    t_ub,
    ksd,
    m2,
    m3,
    M2,
):
    upper_bd = (
        spy_stats.norm.cdf(n * (ksd - t_ub) / (2 * M2)**0.5)
    )

    lower_bd = 1 - (
        spy_stats.norm.cdf(n * (t_lb - ksd) / (2 * M2)**0.5)
    )

    return lower_bd, upper_bd

def compute_analytical_new_cond_bounds(
    n,
    t_lb,
    t_ub,
    ksd,
    m2,
    m3,
    M2,
):
    upper_bd = (
        spy_stats.norm.cdf(np.sqrt(n) * (ksd - t_ub) / (2 * m2**0.5))
    )

    lower_bd = 1 - (
        spy_stats.norm.cdf(np.sqrt(n) * (t_lb - ksd) / (2 * m2**0.5))
    )

    return lower_bd, upper_bd

def compute_analytical_new_sum_bounds(
    n,
    t_lb,
    t_ub,
    ksd,
    m2,
    m3,
    M2,
):
    upper_bd = (
        spy_stats.norm.cdf(np.sqrt(n) * (ksd - t_ub) / np.sqrt(4 * m2 + 2 * M2 / n))
    )

    lower_bd = 1 - (
        spy_stats.norm.cdf(np.sqrt(n) * (t_lb - ksd) / np.sqrt(4 * m2 + 2 * M2 / n))
    )

    return lower_bd, upper_bd

def power_experiment(
    ksd_res,
    res_analytical,
    ns,
    dims,
    ts_lb,
    ts_ub,
    bound: str="markov",
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
        t_lb = ts_lb[i]
        t_ub = ts_ub[i]
        
        # compute analytical bounds
        if bound == "markov":
            l_bd, u_bd = compute_analytical_markov_bounds(
                n,
                t_lb,
                t_ub,
                ksd=res_analytical["ksd"][i],
                m2=res_analytical["cond_var"][i],
                M2=res_analytical["full_var"][i],
            )

        elif bound == "be":
            l_bd, u_bd = compute_analytical_BE_bounds(
                n,
                t_lb,
                t_ub,
                ksd=res_analytical["ksd"][i],
                m2=res_analytical["cond_var"][i],
                m3=res_analytical["m3"][i],
                M2=res_analytical["full_var"][i],
            )
        
        
        res["l_bd"].append(l_bd)
        res["u_bd"].append(u_bd)
        
        # choose t
        res["t_l"].append(t_lb)
        res["t_u"].append(t_ub)

        # compute empirical probs
        ksd_vals = ksd_res[d]
        res["probs_l"].append(np.mean(ksd_vals >= t_lb))
        res["probs_u"].append(np.mean(ksd_vals >= t_ub))

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

        elif bound == "new_full":
            l_bd, u_bd = compute_analytical_new_full_bounds(
                n,
                t_lb,
                t_ub,
                ksd=res_analytical["ksd"],
                m2=res_analytical["cond_var"],
                m3=res_analytical["m3"],
                M2=res_analytical["full_var"],
            )

        elif bound == "new_cond":
            l_bd, u_bd = compute_analytical_new_cond_bounds(
                n,
                t_lb,
                t_ub,
                ksd=res_analytical["ksd"],
                m2=res_analytical["cond_var"],
                m3=res_analytical["m3"],
                M2=res_analytical["full_var"],
            )

        elif bound == "new_sum":
            l_bd, u_bd = compute_analytical_new_sum_bounds(
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

def bootstrap_quantile(
    dims,
    ns,
    num_boot,
    nreps,
    kernel_class,
    bandwidth_order,
    delta: float=2.,
):
    """Not in use!"""
    assert ValueError("Not in use!")
    res = {d: [] for d in dims}

    iterator = trange(len(ns))
    for i in iterator:
        n = ns[i]
        d = dims[i]

        target_dist, sample_dist = generate_target_proposal(d, delta)
        X = sample_dist.sample((nreps, n))
        
        # initialise KSD
        kernel = kernel_class(sigma_sq=2.*d**bandwidth_order)
        ksd = KSD(kernel=kernel, log_prob=target_dist.log_prob)
        
        bootstrap = Bootstrap(ksd, n)
        # (num_boot - 1) because the test statistic is also included 
        multinom_samples = bootstrap.multinom.sample((nreps, num_boot-1,)) # num_boot x ntest
        
        for j in range(nreps):
            iterator.set_description(f"[{j+1} / {nreps}]")

            multinom_sample_j = multinom_samples[j]
            X_j = X[j]

            u_p = bootstrap.compute_test_statistic(X_j)

            bootstrap.compute_bootstrap(num_boot=num_boot, u_p=u_p, multinom_samples=multinom_sample_j)

            _, q, _ = bootstrap._test_once(alpha=0.05)

            res[d].append(q)

    return res
    

class LimitDistExperiment:
    def __init__(
        self,
        empirical_vals,
        res_analytical,
        n,
        ts,
    ):
        self.empirical_vals = empirical_vals 
        self.res_analytical = res_analytical
        self.n = n
        self.ts = ts
        
        self.val = self.res_analytical["expectation"]
        self.m2 = self.res_analytical["cond_var"]
        self.M2 = self.res_analytical["full_var"]

    def compute_empirical_prob(self, t):
        return np.mean(self.empirical_vals >= t)

    def compute_normal_cond_prob(self, t):
        res = 1 - spy_stats.norm.cdf(
            np.sqrt(self.n) * (t - self.val) / (2 * self.m2**0.5)
        )
        return res

    def compute_normal_full_prob(self, t):
        res = 1 - spy_stats.norm.cdf(
            self.n * (t - self.val) / (2 * self.M2)**0.5
        )
        return res

    def compute_normal_sum_prob(self, t):
        res = 1 - spy_stats.norm.cdf(
            np.sqrt(self.n) * (t - self.val) / np.sqrt(4 * self.m2 + 2 * self.M2 / self.n)
        )
        return res

    def compute_mm_full_prob(self, t):
        var_Dn = 2 * self.M2 / (self.n * (self.n - 1))
        alpha = self.val**2 / var_Dn
        beta = self.val / var_Dn
        res = 1 - spy_stats.gamma.cdf(
            t,
            a=alpha,
            scale=1/beta,
        )
        return res

    def compute_mm_chi_prob(self, t):
        var_Dn = 2 * self.M2 / (self.n * (self.n - 1))
        res = 1 - spy_stats.chi2.cdf(
            np.sqrt(2. / var_Dn) * (t - self.val) + 1,
            df=1,
        )
        return res

    def run(self, bound):
        """
        Plot power and bounds as a function of decision threshold t.
        """
        res = {
            "bound": [], #[bound] * len(self.ts),
            "n": [], #[self.n] * len(self.ts),
            "probs": [],
        }
        
        for t in self.ts:
            if bound == "probs":
                # compute empirical probs
                res["probs"].append(self.compute_empirical_prob(t))
                res["bound"].append("empirical")

            elif bound == "cond":
                # compute gaussian cond var
                res["probs"].append(self.compute_normal_cond_prob(t))
                res["bound"].append("cond")

            elif bound == "full":
                # compute gaussian full var
                res["probs"].append(self.compute_normal_full_prob(t))
                res["bound"].append("full")

            elif bound == "sum":
                # compute gaussian sum var
                res["probs"].append(self.compute_normal_sum_prob(t))
                res["bound"].append("sum")

            elif bound == "mm_full":
                # compute moment matching full var
                res["probs"].append(self.compute_mm_full_prob(t))
                res["bound"].append("mm_full")

            elif bound == "mm_chi":
                # compute moment matching chi-sq
                res["probs"].append(self.compute_mm_chi_prob(t))
                res["bound"].append("mm_chi")

        return res

    def sample_normal_cond(self, size):
        res = np.random.normal(
            loc=self.val,
            scale=2 * self.m2**0.5 / np.sqrt(self.n),
            size=size,
        )
        return res

    def sample_normal_full(self, size):
        res = np.random.normal(
            loc=self.val,
            scale=(2 * self.M2)**0.5 / self.n,
            size=size,
        )
        return res

    def sample_normal_sum(self, size):
        res = np.random.normal(
            loc=self.val,
            scale=np.sqrt(4 * self.m2 + 2 * self.M2 / self.n) / np.sqrt(self.n),
            size=size,
        )
        return res

    def sample_mm_full(self, size):
        var_Dn = 2 * self.M2 / (self.n * (self.n - 1))
        alpha = self.val**2 / var_Dn
        beta = self.val / var_Dn
        res = spy_stats.gamma.rvs(
            a=alpha,
            scale=1/beta,
            size=size
        )
        return res

    def sample_mm_chi(self, size):
        var_Dn = 2 * self.M2 / (self.n * (self.n - 1))
        sigma = np.sqrt(var_Dn / 2.)
        res = spy_stats.chi2.rvs(
            df=1,
            size=size,
        )
        res = sigma * (res - 1) + self.val
        return res

    def sample(self, limit, size: int=None):
        size = len(self.empirical_vals) if size is None else size
        
        if limit == "probs":
            res = self.empirical_vals
        
        elif limit == "cond":
            # sample gaussian cond var
            res = self.sample_normal_cond(size)
        
        elif limit == "full":
            # sample gaussian full var
            res = self.sample_normal_full(size)
        
        elif limit == "sum":
            # sample gaussian sum var
            res = self.sample_normal_sum(size)
        
        elif limit == "mm_full":
            # sample moment matching full var
            res = self.sample_mm_full(size)

        elif limit == "mm_chi":
            # sample moment matching chi-sq
            res = self.sample_mm_chi(size)

        return res

    def compute_tail_prob(self, threshold):
        abs_diff = np.abs(np.array(self.empirical_vals) - self.val)
        return np.mean(abs_diff > threshold)

class LimitDistExperimentRepeated:
    """Single dim, single sample size, multiple repetitions."""
    def __init__(
        self,
        empirical_vals_list,
        res_analytical,
        n,
        ts,
    ):
        self.empirical_vals_list = empirical_vals_list
        self.ts = list(ts)
        self.ts_len = len(ts)
        self.res_analytical = res_analytical

        self.experiments = [
            LimitDistExperiment(
                vals,
                res_analytical,
                n,
                ts,
            )
            for vals in self.empirical_vals_list
        ]

    def run(self):
        res = {
            "value": [],
            "name": [],
            "seed": [],
            "ts": [],
        }
        name_list = ["cond", "full", "sum", "mm_full", "mm_chi"]

        for s, exp in enumerate(self.experiments):
            # compute empirical exceeding probability
            res["value"] += exp.run("probs")["probs"]
            res["name"] += ["probs"] * self.ts_len
            res["seed"] += [s] * self.ts_len
            res["ts"] += self.ts

        for name in name_list:
            # compute cdf of limits
            res["value"] += exp.run(name)["probs"]
            res["name"] += [name] * self.ts_len
            res["seed"] += [s] * self.ts_len
            res["ts"] += self.ts
        
        res = pd.DataFrame(res)
        return res

    def compute_distance(self, size: int=None):
        res = {
            "dist": [],
            "name": [],
            "seed": [],
        }
        name_list = ["cond", "full", "sum", "mm_full", "mm_chi"]

        for s, exp in enumerate(self.experiments):
            Dn = exp.sample("probs")

            for name in name_list:
                # compute cdf of limits
                lim_samples = exp.sample(name, size)
                # dist = spy_stats.wasserstein_distance(Dn, lim_samples)
                # dist = spy_stats.energy_distance(Dn, lim_samples)
                dist = spy_stats.kstest(Dn, lim_samples).statistic
                res["dist"].append(dist)
                res["name"].append(name)
                res["seed"].append(s)

        res = pd.DataFrame(res)
        return res

    def sample_all_limits(self, size: int=None):
        res = {
            "x": [],
            "name": [],
            "seed": [],
        }
        name_list = ["cond", "full", "sum", "mm_full", "mm_chi"]

        for s, exp in enumerate(self.experiments):
            Dn = exp.sample("probs")
            res["x"] += Dn
            res["name"] += ["probs"] * len(Dn)
            res["seed"] += [s] * len(Dn)

            for name in name_list:
                # compute cdf of limits
                lim_samples = exp.sample(name, size).tolist()
                res["x"] += lim_samples
                res["name"] += [name] * len(lim_samples)
                res["seed"] += [s] * len(lim_samples)

        res = pd.DataFrame(res)
        return res

    def compute_tail_prob(self, threshold: float):
        res = {"prob": [], "seed": []}
        for s, exp in enumerate(self.experiments):
            tail_prob = exp.compute_tail_prob(threshold)
            res["prob"].append(tail_prob)
            res["seed"].append(s)

        res = pd.DataFrame(res)
        return res

class LimitDistExperimentRepeatedMultiDims:
    """Multiple dims, multiple sample sizes, multiple repetitions."""
    def __init__(
        self,
        dims,
        empirical_vals_dims_list,
        res_analytical,
        ns,
        ts,
    ):
        self.dims = dims
        self.empirical_vals_dims_list = empirical_vals_dims_list
        self.ts = list(ts)
        self.ts_len = len(ts)
        self.res_analytical = res_analytical
        self.experiments = [
            LimitDistExperimentRepeated(
                [l[d] for l in empirical_vals_dims_list],
                res_analytical.loc[d].to_dict(),
                ns[i],
                ts,
            )
            for i, d in enumerate(self.dims)
        ]

    def get_exp(self, dim):
        ind = self.dims.index(dim)
        return self.experiments[ind]

    def run(self, dim):
        exp = self.get_exp(dim)
        return exp.run()

    def compute_distance(self, size: int=None):
        res_list = []
        
        iterator = tqdm(enumerate(self.experiments), total=len(self.experiments))
        for i, exp in iterator:
            res = exp.compute_distance(size)
            res["dim"] = self.dims[i]
            res_list.append(res)

        res = pd.concat(res_list, ignore_index=True)
        return res

    def compute_tail_prob(self, threshold: float):
        res_list = []

        for i, exp in enumerate(self.experiments):
            res = exp.compute_tail_prob(threshold)
            res["dim"] = self.dims[i]
            res_list.append(res)

        res = pd.concat(res_list, ignore_index=True)

        return res
