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
from experiments.compare_samplers_convolved_med import var_med_heuristic

tf.random.set_seed(0)

def run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, kernel, alpha, num_boot, num_est):
    """compute KSD and repeat for nrep times"""
    ksd = ConvolvedKSD(target=target, kernel=kernel, conv_kernel=None) #TODO conv_kernel is not used in class
    
    n = 500
    ksd_df = pd.DataFrame(columns=["n", "p_value", "var_est", "seed", "type"])
    iterator = trange(nrep)
    bootstrap = Bootstrap(ksd, n)
    multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x n
    print(f"Running with sample size {n}")
    for seed in iterator:
        iterator.set_description(f"Repetition: {seed+1} of {nrep}")
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

        ksd = ConvolvedKSD(target=target, kernel=kernel, conv_kernel=None)
        bootstrap = Bootstrap(ksd, n)

        # off-target sample without noise
        _, p_val = bootstrap.test_once(
            alpha=alpha, 
            num_boot=num_boot, 
            X=proposal_off_sample, 
            multinom_samples=multinom_samples[seed, :], 
            conv_samples_full=conv_sample_full,
            conv_samples=conv_sample,
            log_noise_std=-1e18)
        ksd_df.loc[len(ksd_df)] = [n, p_val, 0, seed, "off-target noiseless"]

        # on-target sample without noise
        _, p_val = bootstrap.test_once(
            alpha=alpha, 
            num_boot=num_boot, 
            X=proposal_on_sample, 
            multinom_samples=multinom_samples[seed, :], 
            conv_samples_full=conv_sample_full,
            conv_samples=conv_sample,
            log_noise_std=-1e18)
        ksd_df.loc[len(ksd_df)] = [n, p_val, 0, seed, "target noiseless"]

        # off-target sample
        proposal_off_sample += conv_sample
        _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=proposal_off_sample, multinom_samples=multinom_samples[seed, :], conv_samples=conv_sample_full)
        ksd_df.loc[len(ksd_df)] = [n, p_val, var.numpy(), seed, "off-target"]

        # on-target sample
        proposal_on_sample += conv_sample
        _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=proposal_on_sample, multinom_samples=multinom_samples[seed, :], conv_samples=conv_sample_full)
        ksd_df.loc[len(ksd_df)] = [n, p_val, var.numpy(), seed, "target"]
    return ksd_df

parser = argparse.ArgumentParser()
nrep = 1000
num_boot = 1000 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
delta_list = [1., 2., 4., 6.]
dim = 5
num_est = 10000 # num samples used to estimate concolved target
parser.add_argument("--load", type=str, default="", help="path to pre-saved results")
args = parser.parse_args()
args.load = "res/bootstrap"
print(args.load)

if __name__ == '__main__':
    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 9))
    subfigs = fig.subfigures(1, len(delta_list))
    for ind, delta in enumerate(delta_list):
        print(f"Running with delta = {delta}")
        test_imq_df = None

        # target distribution
        target = create_mixture_gaussian(dim=dim, delta=delta)

        # off-target proposal distribution
        proposal_on = create_mixture_gaussian(dim=dim, delta=delta)
        
        # off-target proposal distribution
        proposal_mean = - delta * tf.eye(dim)[:, 0]
        proposal_off = tfd.MultivariateNormalDiag(proposal_mean)

        if len(args.load) > 0 :
            try:
                test_imq_df = pd.read_csv(args.load + f"/med_delta{delta}.csv")
                print(f"Loaded pre-saved data for delta={delta}")
            except:
                print(f"Pre-saved data for delta={delta} not found. Running from scratch now.")

        if test_imq_df is None:
            # with IMQ
            imq = IMQ(med_heuristic=True)
            test_imq_df = run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, imq, alpha, num_boot, num_est)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(4, 1)
        axs = axs.flat

        var = tf.constant(test_imq_df.loc[test_imq_df.n == test_imq_df.n.max(), "var_est"].median(), dtype=tf.float32)
        convolution = tfd.MultivariateNormalDiag(0., tf.math.sqrt(var) * tf.ones(dim))
        convolution_sample = convolution.sample(10000)
        axs[0].hist((proposal_off.sample(10000) + convolution_sample).numpy()[:, 0], label="off-target", alpha=0.2)
        axs[0].hist((proposal_on.sample(10000) + convolution_sample).numpy()[:, 0], label="target", alpha=0.2)
        axs[0].legend()

        # sns.histplot(ax=axs[1], data=test_imq_df.loc[test_imq_df.type == "off-target"], x="p_value", hue="type", bins=20)
        sns.ecdfplot(ax=axs[1], data=test_imq_df.loc[test_imq_df.type.isin(["off-target", "off-target noiseless"])], x="p_value", hue="type")
        axs[1].plot([0, 1], [0, 1], transform=axs[1].transAxes, color="grey", linestyle="dashed")
        axs[1].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        err = (test_imq_df.loc[test_imq_df.type == "off-target", "p_value"] > alpha).mean()
        axs[1].set_title(f"off target (type II error = {err})")
        axs[1].set_xlabel("p-value")
        
        # sns.histplot(ax=axs[2], data=test_imq_df.loc[test_imq_df.type == "target"], x="p_value", hue="type", bins=20)
        sns.ecdfplot(ax=axs[2], data=test_imq_df.loc[test_imq_df.type.isin(["target", "target noiseless"])], x="p_value", hue="type")
        axs[2].plot([0, 1], [0, 1], transform=axs[2].transAxes, color="grey", linestyle="dashed")
        axs[2].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        axs[2].axis(xmin=-0.01, xmax=1.)
        err = (test_imq_df.loc[test_imq_df.type == "target", "p_value"] <= alpha).mean()
        axs[2].set_title(f"On target (type I error = {err})")
        axs[2].set_xlabel("p-value")

        sns.histplot(ax=axs[3], data=test_imq_df.loc[test_imq_df.type == "target"], x="var_est", bins=20)
        axs[3].set_title("IMQ var estimates of noise")

        # save res
        test_imq_df.to_csv(f"res/bootstrap/med_delta{delta}.csv", index=False)

    # plt.tight_layout()
    fig.savefig("figs/bootstrap/bootstrap_convolved_med.png")
