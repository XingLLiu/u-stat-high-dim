from operator import index
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import argparse

from src.ksd.ksd import ConvolvedKSD
from src.ksd.kernel import RBF, IMQ
from src.ksd.bootstrap import Bootstrap
from experiments.compare_samplers import create_mixture_gaussian

tf.random.set_seed(0)

def run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, convolution, kernel, alpha, num_boot, num_est):
    """compute KSD and repeat for nrep times"""
    ksd = ConvolvedKSD(target=target, kernel=kernel, conv_kernel=convolution)
    
    n = 500
    ksd_df = pd.DataFrame(columns=["n", "p_value", "seed", "type"])
    iterator = trange(nrep)
    bootstrap = Bootstrap(ksd, n)
    multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x n
    iterator.set_description(f"Running with sample size {n}")
    for seed in iterator:
        iterator.set_description(f"Repetition: {seed+1} of {nrep}")
        # convolution sample
        conv_sample_full = convolution.sample(num_est) # for p

        conv_ind = tf.experimental.numpy.random.randint(low=0, high=num_est, size=n)
        conv_sample = tf.gather(conv_sample_full, conv_ind, axis=0) # for q
        
        # off-target sample
        proposal_off_sample = proposal_off.sample(n)
        proposal_off_sample += conv_sample
        _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=proposal_off_sample, multinom_samples=multinom_samples[seed, :], conv_samples=conv_sample_full)
        ksd_df.loc[len(ksd_df)] = [n, p_val, seed, "off-target"]

        # on-target sample
        proposal_on_sample = proposal_on.sample(n)
        proposal_on_sample += conv_sample
        _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=proposal_on_sample, multinom_samples=multinom_samples[seed, :], conv_samples=conv_sample_full)
        ksd_df.loc[len(ksd_df)] = [n, p_val, seed, "target"]
    return ksd_df

class TruncatedDistribution:
    def __init__(self, dist, low, high):
        """truncate the **first** coordinate to [low, high].
        
        Inputs:
            low, high: scalars
        """
        self.dist = dist
        self.low = low
        self.high = high
    def prob(self, x):
        """prob upto normalizing const"""
        return self.dist.prob(x)
    def sample(self, size):
        x = self.dist.sample(size*5)
        samples = tf.boolean_mask(x, (x[:, 0] >= self.low) & (x[:, 0] <= self.high))
        n = samples.shape[0]
        while n < size:
            x = self.dist.sample(size) # size x dim
            samples = tf.concat([samples, tf.boolean_mask(x, (x[:, 0] >= self.low) & (x[:, 0] <= self.high))], axis=0)
            n = samples.shape[0]
        return samples[:size, :]


parser = argparse.ArgumentParser()
nrep = 1000
num_boot = 1000 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
delta = 4.0
var_list = [1e-4, 1e-2, 1., 5., 10.] # [1e-2, 0.1, 1., 5., 10., 20., 30., 40., 50., 100., ]
dim = 5
num_est = 10000 # num samples used to estimate concolved target
parser.add_argument("--load", type=str, default="")
args = parser.parse_args()

if __name__ == '__main__':
    fig = plt.figure(constrained_layout=True, figsize=(5*len(var_list), 9))
    subfigs = fig.subfigures(1, len(var_list))
    for ind, var in enumerate(var_list):
        print(f"Running with var = {var}")
        test_imq_df = None

        # target distribution
        target = create_mixture_gaussian(dim=dim, delta=delta)

        # convolution kernel
        convolution = tfd.MultivariateNormalDiag(0., tf.math.sqrt(var) * tf.ones(dim))

        # off-target proposal distribution
        proposal_on = create_mixture_gaussian(dim=dim, delta=delta)
        
        # off-target proposal distribution
        proposal_mean = - delta * tf.eye(dim)[:, 0]
        proposal_off = TruncatedDistribution(dist=proposal_on, low=proposal_mean[0]*2, high=0.)

        if len(args.load) > 0 :
            try:
                test_imq_df = pd.read_csv(args.load + f"/trunc_convolved_var{var}.csv")
                print(f"Loaded pre-saved data for var={var}")
            except:
                print(f"Pre-saved data for var={var} not found. Running from scratch now.")

        if test_imq_df is None:
            # with IMQ
            imq = IMQ(med_heuristic=True)
            test_imq_df = run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, convolution, imq, alpha, num_boot, num_est)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"var = {var}")
        axs = subfig.subplots(3, 1)
        axs = axs.flat
        convolution_sample = convolution.sample(10000)
        axs[0].hist((proposal_off.sample(10000) + convolution_sample).numpy()[:, 0], label="off-target", alpha=0.2)
        axs[0].hist((proposal_on.sample(10000) + convolution_sample).numpy()[:, 0], label="target", alpha=0.2)
        axs[0].legend()

        sns.histplot(ax=axs[1], data=test_imq_df.loc[test_imq_df.type == "off-target"], x="p_value", hue="type", bins=20)
        axs[1].axis(xmin=-0.01, xmax=1.)
        err = (test_imq_df.loc[test_imq_df.type == "off-target", "p_value"] > alpha).mean()
        axs[1].set_title(f"off target (type II error = {err})")
        axs[1].set_xlabel("p-value")
        
        sns.histplot(ax=axs[2], data=test_imq_df.loc[test_imq_df.type == "target"], x="p_value", hue="type", bins=20)
        axs[2].axis(xmin=-0.01, xmax=1.)
        err = (test_imq_df.loc[test_imq_df.type == "target", "p_value"] <= alpha).mean()
        axs[2].set_title(f"On target (type I error = {err})")
        axs[2].set_xlabel("p-value")

        # save res
        test_imq_df.to_csv(f"res/bootstrap/trunc_convolved_var{var}.csv", index=False)

    # plt.tight_layout()
    fig.savefig("figs/bootstrap_convolved_truncated.png")
