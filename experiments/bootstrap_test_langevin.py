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

from src.ksd.ksd import KSD
from src.ksd.kernel import RBF, IMQ
from src.ksd.bootstrap import Bootstrap
from src.ksd.models import create_mixture_gaussian
from src.ksd.langevin import Langevin

tf.random.set_seed(0)

def run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, kernel, alpha, num_boot, t_list, step_size):
    """compute KSD and repeat for nrep times"""
    ksd = KSD(target=target, kernel=kernel)
    
    n = 500
    T = int(max(t_list)) + 1 # maximal number of steps

    ksd_df = pd.DataFrame(columns=["n", "t", "p_value", "seed", "type"])
    iterator = trange(nrep)
    bootstrap = Bootstrap(ksd, n)
    multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x n
    iterator.set_description(f"Running with sample size {n}")
    for seed in iterator:
        iterator.set_description(f"Repetition: {seed+1} of {nrep}")
        # run stochastic process on off-target sample for T steps
        off_sample_init = proposal_off.sample(n)
        langevin_off = Langevin(log_prob=proposal_on.log_prob)
        langevin_off.run(steps=T, step_size=step_size, x_init=off_sample_init)

        # run stochastic process on off-target sample for T steps
        on_sample_init = proposal_on.sample(n)
        langevin_on = Langevin(log_prob=proposal_on.log_prob)
        langevin_on.run(steps=T, step_size=step_size, x_init=on_sample_init)
        
        for t in t_list:
            # off-target sample
            off_sample = langevin_off.x[t, :, :]
            _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=off_sample, multinom_samples=multinom_samples[seed, :])
            ksd_df.loc[len(ksd_df)] = [n, t, p_val, seed, "off-target"]

            # on-target sample
            on_sample = langevin_on.x[t, :, :]
            _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=on_sample, multinom_samples=multinom_samples[seed, :])
            ksd_df.loc[len(ksd_df)] = [n, t, p_val, seed, "target"]

    return ksd_df

parser = argparse.ArgumentParser()
nrep = 1000
num_boot = 1000 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
delta = 4.0
t_list = [0, 100, 500, 1000]
step_size = 1e-3 # step size for Langevin dynamic
dim = 5
num_est = 10000 # num samples used to estimate concolved target
parser.add_argument("--load", type=str, default="")
args = parser.parse_args()

if __name__ == '__main__':
    fig = plt.figure(constrained_layout=True, figsize=(5*len(t_list), 9))
    subfigs = fig.subfigures(1, len(t_list))
    
    print(f"Running with step size = {step_size}")
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
            test_imq_df = pd.read_csv(args.load + f"/langevin_stepsize{step_size}.csv")
            print(f"Loaded pre-saved data for step size = {step_size}")
        except:
            print(f"Pre-saved data for step size = {step_size} not found. Running from scratch now.")

    if test_imq_df is None:
        # with IMQ
        imq = IMQ(med_heuristic=True)
        test_imq_df = run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, imq, alpha, num_boot, t_list, step_size)

        # save res
        test_imq_df.to_csv(f"res/bootstrap/langevin_stepsize{step_size}.csv", index=False)


    # for plotting dist of Langevin-perturbed samples
    off_sample = proposal_off.sample(10000)
    on_sample = proposal_on.sample(10000)
    langevin_off = Langevin(log_prob=proposal_on.log_prob)
    langevin_off.run(steps=int(max(t_list))+1, step_size=step_size, x_init=off_sample)

    langevin_on = Langevin(log_prob=proposal_on.log_prob)
    langevin_on.run(steps=int(max(t_list))+1, step_size=step_size, x_init=on_sample)

    for ind, t in enumerate(t_list):
        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"step = {t}")
        axs = subfig.subplots(4, 1)
        axs = axs.flat

        axs[0].hist(langevin_off.x[t, :, :].numpy()[:, 0], label="perturbed off-target", alpha=0.2, bins=40)
        axs[0].hist(on_sample.numpy()[:, 0], label="on-target", alpha=0.2, bins=40)
        axs[0].hist(off_sample.numpy()[:, 0], label="off-target", alpha=0.2, bins=40)
        axs[0].legend()

        samples_df_target = pd.DataFrame({"x1": on_sample.numpy()[:, 0], "type": "target"})
        samples_df_perturbed = pd.DataFrame({"x1": langevin_on.x[t, :, :].numpy()[:, 0], "type": "perturbed target"})
        samples_df = pd.concat([samples_df_target, samples_df_perturbed], ignore_index=True)
        # sns.ecdfplot(ax=axs[1], x=langevin_on.x[t, :, :].numpy()[:, 0], label="perturbed target")
        # sns.ecdfplot(ax=axs[1], x=on_sample.numpy()[:, 0], label="target")
        sns.ecdfplot(ax=axs[1], data=samples_df, x="x1", hue="type")
        axs[1].set_ylabel("CDF")

        sns.ecdfplot(ax=axs[2], data=test_imq_df.loc[(test_imq_df.type == "off-target") & (test_imq_df.t == t)], x="p_value", hue="type")
        axs[2].plot([0, 1], [0, 1], transform=axs[2].transAxes, color="grey", linestyle="dashed")
        axs[2].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        err = (test_imq_df.loc[test_imq_df.type == "off-target", "p_value"] > alpha).mean()
        axs[2].set_title(f"off target (type II error = {err})")
        axs[2].set_xlabel("p-value")
        
        sns.ecdfplot(ax=axs[3], data=test_imq_df.loc[(test_imq_df.type == "target") & (test_imq_df.t == t)], x="p_value", hue="type")
        axs[3].plot([0, 1], [0, 1], transform=axs[3].transAxes, color="grey", linestyle="dashed")
        axs[3].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        err = (test_imq_df.loc[test_imq_df.type == "target", "p_value"] <= alpha).mean()
        axs[3].set_title(f"On target (type I error = {err})")
        axs[3].set_xlabel("p-value")


    # plt.tight_layout()
    fig.savefig("figs/bootstrap/bootstrap_langevin.png")
