import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.ksd.ksd import KSD
from src.ksd.kernel import RBF, IMQ

tf.random.set_seed(0)

def run_ksd_experiment(nrep, target, proposal, kernel):
    """compute KSD and repeat for nrep times"""
    ksd = KSD(target=target, kernel=kernel)
    
    nsamples_list = list(range(10, 100, 10)) + list(range(100, 1100, 100))
    ksd_list = []
    ksd_df = pd.DataFrame(columns=["n", "ksd", "seed", "type"])
    for n in nsamples_list:
        print(f"n = {n}")
        for seed in range(nrep):
            # off-target sample
            proposal_sample = proposal.sample(n)
            ksd_val = ksd(proposal_sample, tf.identity(proposal_sample)).numpy()
            ksd_df.loc[len(ksd_df)] = [n, ksd_val, seed, "off-target"]

            # on-target sample
            proposal_sample = target.sample(n)
            ksd_val = ksd(proposal_sample, tf.identity(proposal_sample)).numpy()
            ksd_df.loc[len(ksd_df)] = [n, ksd_val, seed, "target"]
    return ksd_df

def create_mixture_gaussian(dim, delta):
    e1 = tf.eye(dim)[:, 0]
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[0.5, 0.5]),
      components=[
        tfd.MultivariateNormalDiag(-delta * e1),
        tfd.MultivariateNormalDiag(delta * e1)
    ])
    return mix_gauss
    

nrep = 10
delta_list = [0.5, 1.5, 3.0]
dim = 5

subfigs = plt.subfigures(1, len(delta_list))
for ind, delta in enumerate(delta_list):
    # target distribution
    target = create_mixture_gaussian(dim=dim, delta=delta)

    # proposal distribution
    proposal_mean = - delta * tf.eye(dim)[:, 0]
    proposal = tfd.MultivariateNormalDiag(proposal_mean)

    # with IMQ
    imq = IMQ()
    ksd_imq_df = run_ksd_experiment(nrep, target, proposal, imq)

    # with RBF
    rbf = RBF()
    ksd_rbf_df = run_ksd_experiment(nrep, target, proposal, rbf)

    # plot
    subfig = subfigs.flat[ind]
    fig, axs = subfig.subplots(2, 1)
    sns.lineplot(ax=axs[0, 0], data=ksd_imq_df, x="n", y="ksd", hue="type")
    _ = plt.ylim((0, None))
    axs[0, 0].set_title("IMQ")
    plt.xscale("log")
    
    sns.lineplot(ax=axs[0, 1], data=ksd_rbf_df, x="n", y="ksd", hue="type")
    _ = plt.ylim((0, None))
    axs[0, 0].set_title("RBF")
    plt.xscale("log")

subfigs.savefig("figs/mixture_gaussian.png")