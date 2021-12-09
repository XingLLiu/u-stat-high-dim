import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

from src.ksd.ksd import ConvolvedKSD
from src.ksd.kernel import RBF, IMQ
from experiments.compare_samplers import create_mixture_gaussian

tf.random.set_seed(0)

def run_ksd_experiment(nrep, target, proposal_on, proposal_off, kernel, num_est, noptim_steps):
    """compute KSD and repeat for nrep times"""
    # convolution kernel
    convolution = tfd.MultivariateNormalDiag(0., tf.ones(dim))

    ksd = ConvolvedKSD(target=target, kernel=kernel, conv_kernel=convolution)
    
    optimizer = tf.optimizers.Adam(learning_rate=0.1)

    nsamples_list = [10, 20, 40, 60, 80] + list(range(100, 1000, 100)) + list(range(1000, 4000, 1000))
    ksd_df = pd.DataFrame(columns=["n", "ksd", "var_est", "seed", "type"])
    iterator = tqdm(nsamples_list)
    for n in iterator:
        # num train samples for finding sigma
        ntrain = int(n * 0.2)

        for seed in range(nrep):
            iterator.set_description(f"Repetition {seed} of {nrep}")
            # convolution sample
            conv_sample_full = convolution.sample(num_est) # for p

            conv_ind = tf.experimental.numpy.random.randint(low=0, high=num_est, size=n)
            conv_sample = tf.gather(conv_sample_full, conv_ind, axis=0) # for q

            # off-target sample
            proposal_off_sample = proposal_off.sample(n)

            log_noise_std = tf.Variable(1.) #TODO start from 0 instead
            off_sample_train, off_sample_test = proposal_off_sample[:ntrain, :], proposal_off_sample[ntrain:, :]
            conv_sample_train, conv_sample_test = conv_sample[:ntrain, :], conv_sample[ntrain:, :]
            ksd.optim(noptim_steps, log_noise_std, off_sample_train, tf.identity(off_sample_train), conv_sample_full, conv_sample_train, optimizer)

            ksd_val = ksd.eval(log_noise_std=log_noise_std.numpy(), X=off_sample_test, Y=tf.identity(off_sample_test), conv_samples_full=conv_sample_full, conv_samples=conv_sample_test).numpy()
            ksd_df.loc[len(ksd_df)] = [n, ksd_val, tf.exp(2*log_noise_std).numpy(), seed, "off-target"]

            # on-target sample
            proposal_on_sample = proposal_on.sample(n)

            log_noise_std_on = tf.Variable(1.) #TODO start from 0 instead
            on_sample_train, on_sample_test = proposal_on_sample[:ntrain, :], proposal_on_sample[ntrain:, :]
            ksd.optim(noptim_steps, log_noise_std_on, on_sample_train, tf.identity(on_sample_train), conv_sample_full, conv_sample_train, optimizer)

            ksd_val = ksd.eval(log_noise_std=log_noise_std_on.numpy(), X=on_sample_test, Y=tf.identity(on_sample_test), conv_samples_full=conv_sample_full, conv_samples=conv_sample_test).numpy()
            ksd_df.loc[len(ksd_df)] = [n, ksd_val, tf.exp(2*log_noise_std_on).numpy(), seed, "target"]
    return ksd_df


nrep = 20
delta_list = [1., 2., 4., 6.]
mean = 0.
dim = 5
num_est = 10000 # num samples used to estimate concolved target
noptim_steps = 100
ratio = 0.5 #TODO make this an arg

if __name__ == '__main__':
    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 9))
    subfigs = fig.subfigures(1, len(delta_list))
    for ind, delta in enumerate(delta_list):
        print(f"Running with delta = {delta}")
        # target distribution
        target = create_mixture_gaussian(dim=dim, delta=delta)

        # off-target proposal distribution
        proposal_mean = - delta * tf.eye(dim)[:, 0]
        proposal_off = tfd.MultivariateNormalDiag(proposal_mean)

        # on-target proposal distribution
        proposal_on = create_mixture_gaussian(dim=dim, delta=delta)

        # with IMQ
        imq = IMQ()
        ksd_imq_df = run_ksd_experiment(nrep, target, proposal_on, proposal_off, imq, num_est, noptim_steps)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(4, 1)
        axs = axs.flat

        var = tf.constant(ksd_imq_df.loc[(ksd_imq_df.n == ksd_imq_df.n.max()) & (ksd_imq_df.type == "off-target"), "var_est"].mean(), dtype=tf.float32)
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
        
        sns.lineplot(ax=axs[2], data=ksd_imq_df.loc[ksd_imq_df.type == "off-target"], x="n", y="var_est", style="type", markers=True)
        axs[2].set_title("Var estimates for off-target samples")
        axs[2].set_xscale("log")
        axs[2].set_yscale("log")

        sns.lineplot(ax=axs[3], data=ksd_imq_df.loc[ksd_imq_df.type == "target"], x="n", y="var_est", style="type", markers=True)
        axs[3].set_title("Var estimates for on-target samples")
        axs[3].set_xscale("log")
        axs[3].set_yscale("log")

        # save res
        ksd_imq_df.to_csv(f"res/compare_samplers/{delta}_optim.csv", index=False)

    # plt.tight_layout()
    fig.savefig(f"figs/mix_gaussian/dim{dim}_ratio{ratio}_optim.png")