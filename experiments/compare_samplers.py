import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

from src.ksd.ksd import KSD
from src.ksd.kernel import RBF, IMQ

tf.random.set_seed(0)

def run_ksd_experiment(nrep, target, proposal, kernel):
    """compute KSD and repeat for nrep times"""
    ksd = KSD(target=target, kernel=kernel)
    
    nsamples_list = [10, 20, 40, 60, 80] + list(range(100, 1000, 100)) + list(range(1000, 4000, 1000))
    # nsamples_list = [10, 5000]
    ksd_list = []
    ksd_df = pd.DataFrame(columns=["n", "ksd", "seed", "type"])
    for n in tqdm(nsamples_list):
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
delta_list = [0.5, 1.5, 2.5, 3.0]
dim = 5

fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 9))
subfigs = fig.subfigures(1, len(delta_list))
for ind, delta in enumerate(delta_list):
    print(f"Running with delta = {delta}")
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
    subfig.suptitle(f"delta = {delta}")
    axs = subfig.subplots(3, 1)
    axs = axs.flat
    axs[0].hist(proposal.sample(10000).numpy()[:, 0], label="off-target", alpha=0.2)
    axs[0].hist(target.sample(10000).numpy()[:, 0], label="target", alpha=0.2)
    axs[0].legend()

    sns.lineplot(ax=axs[1], data=ksd_imq_df, x="n", y="ksd", hue="type", style="type", markers=True)
    # _ = plt.ylim((0, None))
    # axs[1].axis(ymin=0.)
    axs[1].set_title("IMQ")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    
    sns.lineplot(ax=axs[2], data=ksd_rbf_df, x="n", y="ksd", hue="type", style="type", markers=True)
    # _ = plt.ylim((0, None))
    # axs[2].axis(ymin=0.)
    axs[2].set_title("RBF")
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")

# plt.tight_layout()
fig.savefig("figs/mixture_gaussian.png")