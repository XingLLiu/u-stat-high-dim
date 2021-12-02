import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

from src.ksd.ksd import ConvolvedKSD
from src.ksd.kernel import RBF, IMQ, median_heuristic, l2norm
from experiments.compare_samplers import create_mixture_gaussian

tf.random.set_seed(0)

def var_med_heuristic(samples):
    """Med heuristic for computing the variance of Gaussian noise"""
    dim = samples.shape[1]
    l2norm_mat = l2norm(samples, tf.identity(samples))
    return median_heuristic(l2norm_mat) / dim

def run_ksd_experiment(nrep, target, proposal_on, proposal_off, kernel, num_est):
    """compute KSD and repeat for nrep times"""
    
    nsamples_list = [10, 20, 40, 60, 80] + list(range(100, 1000, 100)) + list(range(1000, 4000, 1000))
    ksd_df = pd.DataFrame(columns=["n", "ksd", "var_est", "seed", "type"])
    for n in tqdm(nsamples_list):
        for seed in range(nrep):
            # draw samples from two samplers
            proposal_off_sample = proposal_off.sample(n)
            proposal_on_sample = proposal_on.sample(n)
            
            # estimate noise var
            var = var_med_heuristic(proposal_on_sample)

            # define noise distribution
            convolution = tfd.MultivariateNormalDiag(0., tf.math.sqrt(var) * tf.ones(dim))
            
            # convolution sample
            conv_sample_full = convolution.sample(num_est) # for p

            conv_ind = tf.experimental.numpy.random.randint(low=0, high=num_est, size=n)
            conv_sample = tf.gather(conv_sample_full, conv_ind, axis=0) # for q

            ksd = ConvolvedKSD(target=target, kernel=kernel, conv_kernel=convolution)
            
            # off-target sample
            # conv_sample = convolution.sample(n)
            proposal_off_sample += conv_sample
            ksd_val = ksd(proposal_off_sample, tf.identity(proposal_off_sample), conv_samples=conv_sample_full).numpy()
            ksd_df.loc[len(ksd_df)] = [n, ksd_val, var.numpy(), seed, "off-target"]

            # on-target sample
            proposal_on_sample += conv_sample
            ksd_val = ksd(proposal_on_sample, tf.identity(proposal_on_sample), conv_samples=conv_sample_full).numpy()
            ksd_df.loc[len(ksd_df)] = [n, ksd_val, var.numpy(), seed, "target"]
    return ksd_df

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

parser = argparse.ArgumentParser()
nrep = 10
delta_list = [1., 2., 4., 6.]
mean = 0.
dim = 5
num_est = 10000 # num samples used to estimate concolved target
parser.add_argument("--ratio", type=float, default="0.5", help="mixture ratio of the off-target")
args = parser.parse_args()

if __name__ == '__main__':
    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 9))
    subfigs = fig.subfigures(1, len(delta_list))
    for ind, delta in enumerate(delta_list):
        print(f"Running with delta = {delta}")
        # target distribution
        target = create_mixture_gaussian(dim=dim, delta=delta, ratio=args.ratio)

        # off-target proposal distribution
        proposal_mean = - delta * tf.eye(dim)[:, 0]
        proposal_off = tfd.MultivariateNormalDiag(proposal_mean)

        # on-target proposal distribution
        proposal_on = create_mixture_gaussian(dim=dim, delta=delta, ratio=args.ratio)

        # with IMQ
        imq = IMQ()
        ksd_imq_df = run_ksd_experiment(nrep, target, proposal_on, proposal_off, imq, num_est)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(3, 1)
        axs = axs.flat

        var = tf.constant(ksd_imq_df.loc[ksd_imq_df.n == ksd_imq_df.n.max(), "var_est"].mean(), dtype=tf.float32)
        convolution = tfd.MultivariateNormalDiag(0., tf.math.sqrt(var) * tf.ones(dim))
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
        
        sns.lineplot(ax=axs[2], data=ksd_imq_df.loc[ksd_imq_df.type == "target"], x="n", y="var_est", style="type", markers=True)
        axs[2].set_title("IMQ var estimates of noise")
        axs[2].set_xscale("log")
        axs[2].set_yscale("log")

    # plt.tight_layout()
    fig.savefig(f"figs/mix_gaussian/dim{dim}_ratio{args.ratio}_med.png")