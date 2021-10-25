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

target = tfd.MultivariateNormalDiag([0.])
proposal = tfd.MultivariateNormalDiag([0.])

# kernel = RBF()
kernel = IMQ()

def run_ksd_experiment(nrep, target, proposal):
    """compute KSD and repeat for nrep times"""
    ksd = KSD(target=target, kernel=kernel)
    
    # nsamples_list = list(range(10, 100, 10)) + list(range(100, 1100, 100))
    nsamples_list = list(range(10, 110, 10))
    ksd_list = []
    ksd_df = pd.DataFrame(columns=["n", "ksd", "seed"])
    for n in nsamples_list:
        print(f"n = {n}")
        ksd_n_list = []
        for seed in range(nrep):
            proposal_sample = proposal.sample(n)
            ksd_val = ksd(proposal_sample, tf.identity(proposal_sample)).numpy()
            ksd_n_list.append(ksd_val)
            ksd_df.loc[len(ksd_df)] = [n, ksd_val, seed]
    return ksd_df

nrep = 10

ksd_df = run_ksd_experiment(nrep, target, proposal)


sns.lineplot(data=ksd_df, x="n", y="ksd")
plt.ylim((0, None))
plt.savefig("figs/rbf_mixture.png")