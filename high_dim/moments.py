from src.ksd.kernel import IMQ, RBF, Linear
import high_dim_power as hd

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--DIR", type=str, default="res/high_dim")
parser.add_argument("--DELTA", type=float, default=0.1,)
parser.add_argument("--R", type=float, default=1., help="bandwidth power")
parser.add_argument("--GAM_SCALE", type=float, default=2., help="bandwidth scale")
parser.add_argument("--STAT", type=str)
parser.add_argument("--KERNEL", type=str)
parser.add_argument("--EXTRA", type=str, default="", help="extra suffix in saved file name")
parser.add_argument("--NPOP", type=int, default=4000)
args = parser.parse_args()

tf.random.set_seed(1)
np.random.seed(1)

DELTA = args.DELTA
BANDWIDTH_POWER = args.R
GAM_SCALE = args.GAM_SCALE
STAT = args.STAT
KERNEL = args.KERNEL
EXTRA = args.EXTRA if args.EXTRA == "" else "_" + args.EXTRA
NPOP = args.NPOP

if KERNEL == "RBF":
    KERNEL = RBF
elif KERNEL == "IMQ":
    KERNEL = IMQ
elif KERNEL == "Linear":
    KERNEL = Linear

SUFFIX = f"delta{DELTA}_r{BANDWIDTH_POWER}_{STAT}_{KERNEL.__name__}{EXTRA}"

if KERNEL == Linear:
    dims = [2, 4, 25, 50, 100, 250, 500, 1000, 2000]

else:
    dims = [1, 2, 4, 25, 50, 100, 250, 500, 1000, 2000]


if EXTRA == "_gamma":
    dims = [50]

elif EXTRA == "_gammaksd":
    dims = [5]
    ns = [50] * len(dims)

if __name__ == "__main__":
    # ground-truth moments
    MOMENTS_DIR = f"{args.DIR}/res_analytical_{SUFFIX}.csv"
    if NPOP != -1:
        res_analytical = hd.compute_population_quantities(
            dims=dims, 
            bandwidth_order=BANDWIDTH_POWER, 
            kernel_class=KERNEL,
            npop=NPOP,
            delta=DELTA,
            statistic=STAT,
            bandwidth_scale=GAM_SCALE,
        )

    else:
        res_analytical = hd.compute_population_quantities_exact(
            dims=dims, 
            bandwidth_order=BANDWIDTH_POWER, 
            kernel_class=KERNEL,
            delta=DELTA,
            statistic=STAT,
            bandwidth_scale=GAM_SCALE,
        )

    # save analytical res
    res_analytical_save = res_analytical
    res_analytical_save["dim"] = dims
    pd.DataFrame(res_analytical_save).to_csv(
        MOMENTS_DIR,
        index=False,
    )

    print("Results saved to:", MOMENTS_DIR)
