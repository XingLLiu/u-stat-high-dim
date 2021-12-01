import numpy as np
from numpy.testing._private.nosetester import _numpy_tester
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
import pickle

from src.ksd.ksd import KSD
from src.ksd.kernel import RBF, IMQ
from src.ksd.bootstrap import Bootstrap
from experiments.compare_samplers import create_mixture_gaussian

tf.random.set_seed(0)

def run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, kernel, alpha, num_boot):
    """compute KSD and repeat for nrep times"""
    ksd = KSD(target=target, kernel=kernel)
    
    # nsamples_list = [10, 20, 40, 60, 80] + list(range(100, 1000, 100)) # + list(range(1000, 4000, 1000))
    nsamples_list = [500]
    ksd_df = pd.DataFrame(columns=["n", "p_value", "seed", "type"])
    iterator = tqdm(nsamples_list)
    for n in iterator:
        bootstrap = Bootstrap(ksd, n)
        multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x n
        iterator.set_description(f"Running with sample size {n}")
        for seed in range(nrep):
            iterator.set_description(f"Repetition: {seed+1} of {nrep}")
            # off-target sample
            proposal_off_sample = proposal_off.sample(n)
            _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=proposal_off_sample, multinom_samples=multinom_samples[seed, :])
            ksd_df.loc[len(ksd_df)] = [n, p_val, seed, "off-target"]

            # on-target sample
            proposal_on_sample = proposal_on.sample(n)
            _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=proposal_on_sample, multinom_samples=multinom_samples[seed, :])
            ksd_df.loc[len(ksd_df)] = [n, p_val, seed, "target"]
    return ksd_df

nrep = 1000
num_boot = 1000 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
delta_list = [1.0, 2.0, 3.0, 4.0]
dim = 5

if __name__ == '__main__':
    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 9))
    subfigs = fig.subfigures(1, len(delta_list))
    for ind, delta in enumerate(delta_list):
        print(f"Running with delta = {delta}")
        # target distribution
        target = create_mixture_gaussian(dim=dim, delta=delta)

        # off-target proposal distribution
        proposal_on = create_mixture_gaussian(dim=dim, delta=delta)
        
        # off-target proposal distribution
        proposal_mean = - delta * tf.eye(dim)[:, 0]
        proposal_off = tfd.MultivariateNormalDiag(proposal_mean)

        # with IMQ
        imq = IMQ(med_heuristic=True)
        test_imq_df = run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, imq, alpha, num_boot)

        # # with RBF
        # rbf = RBF()
        # test_rbf_df = run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, rbf, alpha, num_test, num_boot)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(3, 1)
        axs = axs.flat
        axs[0].hist(proposal_off.sample(10000).numpy()[:, 0], label="off-target", alpha=0.2)
        axs[0].hist(proposal_on.sample(10000).numpy()[:, 0], label="target", alpha=0.2)
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
        # pickle.dump({"imq": test_imq_df, "rbf": test_rbf_df}, open(f"res/bootstrap/delta{delta}", "wb"))
        # pickle.dump({"imq": test_imq_df}, open(f"res/bootstrap/delta{delta}", "wb"))
        # test_imq_df.to_csv(f"res/bootstrap/delta{delta}.csv", index=False)

    # plt.tight_layout()
    fig.savefig("figs/bootstrap.png")
