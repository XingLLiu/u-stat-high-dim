import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

from src.ksd.ksd import KSD, ConvolvedKSD
from src.ksd.kernel import RBF, IMQ
from experiments.compare_samplers import create_mixture_gaussian

tf.random.set_seed(0)

def run_ksd_experiment(nrep, target, proposal_on, proposal_off, convolution, kernel, num_est):
    """compute KSD and repeat for nrep times"""
    ksd = ConvolvedKSD(target=target, kernel=kernel, conv_kernel=convolution)
    # ksd = KSD(target=target, kernel=kernel)
    
    nsamples_list = [10, 20, 40, 60, 80] + list(range(100, 1000, 100)) # + list(range(1000, 4000, 1000))
    ksd_list = []
    ksd_df = pd.DataFrame(columns=["n", "ksd", "seed", "type"])
    for n in tqdm(nsamples_list):
        for seed in range(nrep):
            # off-target sample
            proposal_off_sample = proposal_off.sample(n)
            conv_sample = convolution.sample(n)
            proposal_off_sample += conv_sample
            ksd_val = ksd(proposal_off_sample, tf.identity(proposal_off_sample), num_est).numpy()
            # ksd_val = ksd(proposal_off_sample, tf.identity(proposal_off_sample)).numpy()
            ksd_df.loc[len(ksd_df)] = [n, ksd_val, seed, "off-target"]

            # on-target sample
            proposal_on_sample = proposal_on.sample(n)
            # conv_sample = convolution.sample(n)
            proposal_on_sample += conv_sample
            ksd_val = ksd(proposal_on_sample, tf.identity(proposal_on_sample), num_est).numpy()
            # ksd_val = ksd(proposal_on_sample, tf.identity(proposal_on_sample)).numpy()
            ksd_df.loc[len(ksd_df)] = [n, ksd_val, seed, "target"]
    return ksd_df

def create_convolved_mixture_gaussian(dim, delta, mean, var):
    """
    """
    e1 = tf.eye(dim)[:, 0]
    convolved_scale = tf.math.sqrt(1 + var) * tf.ones(e1.shape[0])
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[0.5, 0.5]),
      components=[
        tfd.MultivariateNormalDiag(-delta * e1 + mean, convolved_scale),
        tfd.MultivariateNormalDiag(delta * e1 + mean, convolved_scale)
    ])
    return mix_gauss

def create_convolved_proposal(dim, proposal_mean, mean, var):
    """
    Convolving with auxiliary variable N(mu, sigma^2) to give
    N(proposal_mean + mean, proposal_var + var).
    This is feasible when only samples are available, as sampling from
    it equivalent to X + Z, where X \sim proposal, and Z \sim auxiliary.
    """
    return tfd.MultivariateNormalDiag(proposal_mean + mean, tf.math.sqrt(1 + var) * tf.ones(proposal_mean.shape[0]))


class MixtureGaussian(tfd.Distribution):
    def __init__(self, delta, dim, *args, **kwargs):
        super(tfd.Distribution, self).__init__(*args, **kwargs)
        self.delta = delta
        e1 = tf.reshape(tf.eye(dim)[:, 0], (1, -1))
        self.mean1 = -delta * e1
        self.mean2 = delta * e1
        self.mix_gauss = tfd.Mixture(
            cat=tfd.Categorical(probs=[0.5, 0.5]),
            components=[
                tfd.MultivariateNormalDiag(tf.reshape(self.mean1, (-1,))),
                tfd.MultivariateNormalDiag(tf.reshape(self.mean2, (-1,)))
        ])
    
    def log_prob(self, x):
        return self.mix_gauss.log_prob(x)

    def grad(self, x):
        """gradient upto normalizing constant"""
        grad1 = - (x - self.mean1) * tf.math.exp(
            - 0.5 * tf.math.reduce_sum((x - self.mean1)**2, axis=1, keepdims=True))
        grad2 = - (x - self.mean2) * tf.math.exp(
            - 0.5 * tf.math.reduce_sum((x - self.mean2)**2, axis=1, keepdims=True))
        return grad1 + grad2

nrep = 10
delta = 4.0
mean = 0.
var_list = [0.01, 0.1, 1., 5., 10., 50., 100.]
dim = 5
num_est = 10000 # num samples used to estimate concolved target

if __name__ == '__main__':
    fig = plt.figure(constrained_layout=True, figsize=(5*len(var_list), 9))
    subfigs = fig.subfigures(1, len(var_list))
    for ind, var in enumerate(var_list):
        print(f"Running with var = {var}")
        # target distribution
        target = create_mixture_gaussian(dim=dim, delta=delta)
        # target = create_convolved_mixture_gaussian(dim=dim, delta=delta, mean=mean, var=var)

        # convolution kernel
        convolution = tfd.MultivariateNormalDiag(0., tf.math.sqrt(var) * tf.ones(dim))

        # off-target proposal distribution
        proposal_mean = - delta * tf.eye(dim)[:, 0]
        # proposal_off = create_convolved_proposal(dim, proposal_mean, mean, var)
        proposal_off = tfd.MultivariateNormalDiag(proposal_mean)

        # on-target proposal distribution
        # proposal_on = create_convolved_mixture_gaussian(dim, delta, mean, var)
        proposal_on = create_mixture_gaussian(dim=dim, delta=delta)

        # with IMQ
        imq = IMQ()
        ksd_imq_df = run_ksd_experiment(nrep, target, proposal_on, proposal_off, convolution, imq, num_est)

        # with RBF
        rbf = RBF()
        ksd_rbf_df = run_ksd_experiment(nrep, target, proposal_on, proposal_off, convolution, rbf, num_est)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"var = {var}")
        axs = subfig.subplots(3, 1)
        axs = axs.flat
        convolution_sample = convolution.sample(10000)
        axs[0].hist((proposal_off.sample(10000) + convolution_sample).numpy()[:, 0], label="off-target", alpha=0.2)
        axs[0].hist((target.sample(10000) + convolution_sample).numpy()[:, 0], label="target", alpha=0.2)
        axs[0].legend()

        sns.lineplot(ax=axs[1], data=ksd_imq_df, x="n", y="ksd", hue="type", style="type", markers=True)
        # _ = plt.ylim((0, None))
        axs[1].axis(ymin=1e-3)
        axs[1].set_title("IMQ")
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        
        sns.lineplot(ax=axs[2], data=ksd_rbf_df, x="n", y="ksd", hue="type", style="type", markers=True)
        # _ = plt.ylim((0, None))
        axs[2].axis(ymin=1e-3)
        axs[2].set_title("RBF")
        axs[2].set_xscale("log")
        axs[2].set_yscale("log")

    # plt.tight_layout()
    fig.savefig("figs/mixture_gaussian_convolved_est.png")