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

def run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, kernel, alpha, num_test, num_boost):
    """compute KSD and repeat for nrep times"""
    ksd = KSD(target=target, kernel=kernel)
    bootstrap = Bootstrap(ksd)
    
    # nsamples_list = [10, 20, 40, 60, 80] + list(range(100, 1000, 100)) # + list(range(1000, 4000, 1000))
    nsamples_list = [10, 20, 50, 100, 250, 500]
    ksd_list = []
    ksd_df = pd.DataFrame(columns=["n", "error_rate", "seed", "type"])
    iterator = tqdm(nsamples_list)
    for n in iterator:
        iterator.set_description(f"Running with sample size {n}")
        for seed in range(nrep):
            # off-target sample
            proposal_off_sample = proposal_off.sample(n)
            # ksd_val = ksd(proposal_off_sample, tf.identity(proposal_off_sample)).numpy()
            _, test_res = bootstrap.test_repeated(alpha=alpha, num_test=num_test, num_boost=num_boost, X=proposal_off_sample, verbose=False)
            err_rate = (np.array(test_res) != 1).sum() / num_test # error if not rejected
            ksd_df.loc[len(ksd_df)] = [n, err_rate, seed, "off-target"]

            # on-target sample
            proposal_on_sample = proposal_on.sample(n)
            # ksd_val = ksd(proposal_on_sample, tf.identity(proposal_on_sample)).numpy()
            _, test_res = bootstrap.test_repeated(alpha=alpha, num_test=num_test, num_boost=num_boost, X=proposal_on_sample, verbose=False)
            err_rate = (np.array(test_res) != 0).sum() / num_test # error if rejected
            ksd_df.loc[len(ksd_df)] = [n, err_rate, seed, "target"]
    return ksd_df

nrep = 1
num_test = 100 # number of tests to calculate the error rate
num_boost = 1000 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
delta_list = [1.0, 4.0] # [1.0, 2.0, 3.0, 4.0]
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
        imq = IMQ()
        test_imq_df = run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, imq, alpha, num_test, num_boost)

        # # with RBF
        # rbf = RBF()
        # test_rbf_df = run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, rbf, alpha, num_test, num_boost)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(2, 1)
        axs = axs.flat
        axs[0].hist(proposal_off.sample(10000).numpy()[:, 0], label="off-target", alpha=0.2)
        axs[0].hist(proposal_on.sample(10000).numpy()[:, 0], label="target", alpha=0.2)
        axs[0].legend()

        sns.lineplot(ax=axs[1], data=test_imq_df, x="n", y="error_rate", hue="type", style="type", markers=True)
        # axs[1].axis(ymin=0.)
        axs[1].set_title("IMQ")
        # axs[1].set_xscale("log")
        # axs[1].set_yscale("log")
        
        # sns.lineplot(ax=axs[2], data=test_rbf_df, x="n", y="error_rate", hue="type", style="type", markers=True)
        # # axs[2].axis(ymin=0.)
        # axs[2].set_title("RBF")
        # # axs[2].set_xscale("log")
        # # axs[2].set_yscale("log")

        # save res
        # pickle.dump({"imq": test_imq_df, "rbf": test_rbf_df}, open(f"res/bootstrap/delta{delta}", "wb"))
        # pickle.dump({"imq": test_imq_df}, open(f"res/bootstrap/delta{delta}", "wb"))
        test_imq_df.to_csv(f"res/bootstrap/delta{delta}.csv", index=False)

    # plt.tight_layout()
    fig.savefig("figs/bootstrap.png")
