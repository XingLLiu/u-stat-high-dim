from src.ksd.find_modes import find_modes, pairwise_directions, run_bfgs
from src.ksd.langevin import RandomWalkMH, RandomWalkBarker
import src.ksd.langevin as mcmc
from src.ksd.ksd import KSD
from src.ksd.kernel import IMQ, l2norm
from src.ksd.bootstrap import Bootstrap
from src.ksd.find_modes import find_modes, pairwise_directions
from src.sensors import plot_sensors

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import pickle
import matplotlib.pyplot as plt
import seaborn as sns


RAM_SCALE_LIST = [0.1, 0.3, 0.5, 0.7, 0.9, 1.08, 1.3]
model_name = "modified_ram"
root_path = "res/sensors/n1000"

loc_true = tf.constant([
    [0.57, 0.91],
    [0.10, 0.37],
    [0.26, 0.14],
    [0.85, 0.04],
#     [0.50, 0.30],
#     [0.30, 0.70]
])

res_df = pd.read_csv(f"{root_path}/res_{model_name}.csv")
ram_scale_list = res_df.ram_scale.tolist()

res_df = res_df[["ram_scale", "p_val_ksd", "p_val_pksd"]].rename(
  {"ram_scale": "RAM scale", "p_val_ksd": "KSD", "p_val_pksd": "pKSD"}
).round(4).T
print(res_df.to_latex(header=False))
