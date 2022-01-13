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
from src.ksd.models import create_mixture_gaussian
from experiments.bootstrap_test_convolved_multiple import run_bootstrap_experiment

tf.random.set_seed(0)

parser = argparse.ArgumentParser()
nrep = 1000
num_boot = 1000 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
delta_list = [1., 2., 4., 6.]
dim = 10
num_est = 10000 # num samples used to estimate concolved target
# grid of noise vars
noise_std_list = [float(2**x) for x in range(-2, 11)]
log_noise_std_list = [tf.math.log(x) for x in noise_std_list]

parser.add_argument("--load", type=str, default="", help="path to pre-saved results")
parser.add_argument("--ratio_t", type=float, default=0.5)
parser.add_argument("--ratio_s", type=float, default=1.)
args = parser.parse_args()
ratio_target = args.ratio_t
ratio_sample = args.ratio_s

if __name__ == '__main__':
    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 9))
    subfigs = fig.subfigures(1, len(delta_list))
    for ind, delta in enumerate(delta_list):
        print(f"Running with delta = {delta}")
        test_imq_df = None

        # define noise distribution
        convolution = tfd.MultivariateNormalDiag(0., tf.ones(dim))

        # target distribution
        target = create_mixture_gaussian(dim=dim, delta=delta, ratio=ratio_target)

        # on-target proposal distribution
        proposal_on = create_mixture_gaussian(dim=dim, delta=delta, ratio=ratio_target)
        
        # off-target proposal distribution
        proposal_off = create_mixture_gaussian(dim=dim, delta=delta, ratio=ratio_sample)

        if len(args.load) > 0 :
            try:
                test_imq_df = pd.read_csv(args.load + f"/multiple_delta{delta}_ratio{ratio_target}_{ratio_sample}.csv")
                print(f"Loaded pre-saved data for delta={delta}")
            except:
                print(f"Pre-saved data for delta={delta} not found. Running from scratch now.")

        if test_imq_df is None:
            # with IMQ
            imq = IMQ(med_heuristic=True)
            test_imq_df = run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, convolution, imq, alpha, num_boot, num_est, log_noise_std_list)

        # save res
        test_imq_df.to_csv(f"res/bootstrap/multiple_delta{delta}_ratio{ratio_target}_{ratio_sample}.csv", index=False)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(5, 1)
        axs = axs.flat

        var = tf.constant(test_imq_df.loc[test_imq_df.type == "off-target", "var_est"].median(), dtype=tf.float32)
        convolution = tfd.MultivariateNormalDiag(0., tf.math.sqrt(var) * tf.ones(dim))
        convolution_sample = convolution.sample(10000).numpy()
        on_sample = proposal_on.sample(10000).numpy()
        off_sample = proposal_off.sample(10000).numpy()
        on_sample_conv = on_sample + convolution_sample
        off_sample_conv = off_sample + convolution_sample
        target_df1 = pd.DataFrame({"x1": on_sample[:, 0], "x2": on_sample[:, 1], "type": "target"}) # noiseless
        target_df2 = pd.DataFrame({"x1": on_sample_conv[:, 0], "x2": on_sample_conv[:, 1], "type": "target"})
        target_df3 = pd.DataFrame({"x1": off_sample[:, 0], "x2": off_sample[:, 1], "type": "off-target"}) # noiseless
        target_df4 = pd.DataFrame({"x1": off_sample_conv[:, 0], "x2": off_sample_conv[:, 1], "type": "off-target"})
        target_df_noiseless = pd.concat([target_df1, target_df3], ignore_index=True)
        target_df = pd.concat([target_df2, target_df4], ignore_index=True)

        sns.histplot(ax=axs[0], data=target_df_noiseless, x="x1", hue="type", alpha=0.3)
        axs[0].set_title("Noiseless target and samples")

        sns.histplot(ax=axs[1], data=target_df, x="x1", hue="type", alpha=0.3)
        axs[1].set_title("convolved with median var over repetitions ({:.3g})".format(var.numpy()))

        sns.ecdfplot(ax=axs[2], data=test_imq_df.loc[test_imq_df.type.isin(["off-target", "off-target noiseless"])], x="p_value", hue="type")
        axs[2].plot([0, 1], [0, 1], transform=axs[2].transAxes, color="grey", linestyle="dashed")
        axs[2].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        err = (test_imq_df.loc[test_imq_df.type == "off-target", "p_value"] > alpha).mean()
        axs[2].set_title(f"off target (type II error = {err})")
        axs[2].set_xlabel("p-value")
        
        sns.ecdfplot(ax=axs[3], data=test_imq_df.loc[test_imq_df.type.isin(["target", "target noiseless"])], x="p_value", hue="type")
        axs[3].plot([0, 1], [0, 1], transform=axs[3].transAxes, color="grey", linestyle="dashed")
        axs[3].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        err = (test_imq_df.loc[test_imq_df.type == "target", "p_value"] <= alpha).mean()
        axs[3].set_title(f"On target (type I error = {err})")
        axs[3].set_xlabel("p-value")

        sns.ecdfplot(ax=axs[4], data=test_imq_df.loc[test_imq_df.type.isin(["off-target", "target"])], x="var_est", hue="type")
        axs[4].axis(ymin=0, ymax=1.01)
        axs[4].set_xscale("log")
        axs[4].set_title("Best var")

    fig.savefig(f"figs/bootstrap/bootstrap_convolved_multiple_ratio{ratio_target}_{ratio_sample}.png")
