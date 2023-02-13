from src.ksd.kernel import IMQ, RBF, Linear
import high_dim_power as hd

import numpy as np
import tensorflow as tf
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--DIR", type=str, default="res/high_dim")
parser.add_argument("--DELTA", type=float, default=2.,)
parser.add_argument("--R", type=float, default=1., help="bandwidth power")
parser.add_argument("--GAM_SCALE", type=float, default=2., help="bandwidth scale")
parser.add_argument("--STAT", type=str)
parser.add_argument("--KERNEL", type=str)
parser.add_argument("--EXTRA", type=str, default="", help="extra suffix in saved file name")
parser.add_argument("--N_EXP", type=int, default=30)
parser.add_argument("--SEED", type=int, default=1)
args = parser.parse_args()

tf.random.set_seed(args.SEED)
np.random.seed(args.SEED)

DELTA = args.DELTA
BANDWIDTH_POWER = args.R
GAM_SCALE = args.GAM_SCALE
STAT = args.STAT
KERNEL = args.KERNEL
EXTRA = args.EXTRA if args.EXTRA == "" else "_" + args.EXTRA
N_EXP = args.N_EXP

if KERNEL == "RBF":
    KERNEL = RBF
elif KERNEL == "IMQ":
    KERNEL = IMQ
elif KERNEL == "Linear":
    KERNEL = Linear

SUFFIX = f"delta{DELTA}_r{BANDWIDTH_POWER}_{STAT}_{KERNEL.__name__}{EXTRA}"

if KERNEL == Linear:
    dims = [2, 4, 25, 50, 100, 250, 500, 1000, 2000]
    ns = [50] * len(dims)
    print("linear kernel!!!")

else:
    dims = [1, 2, 4, 25, 50, 100, 250, 500, 1000, 2000]
    ns = [50] * len(dims)

if EXTRA == "_ld":
    dims = [2] # add back dim 1
    ns = [1000]

elif EXTRA == "_quad":
    # dims = [1, 2, 4, 25, 50, 100, 250, 500, 1000]
    # ns = [max(2, int(d**1.2)) for d in dims]
    dims = [1, 2, 4, 25, 50, 100]
    ns = [max(2, int(d**2)) for d in dims]

elif EXTRA == "_quad100":
    # dims = [1, 2, 4, 25, 50, 100, 250, 500, 1000]
    # ns = [max(2, int(d**1.2)) for d in dims]
    dims = [1, 2, 4, 25, 50, 100]
    ns = [max(2, int(d**2)) for d in dims]

elif EXTRA == "_sqrt":
    dims = [1, 2, 4, 25, 50, 100, 250, 500, 1000, 2000]
    # ns = [25 * int(d**0.5) for d in dims]
    ns = [max(2, int(d**0.5)) for d in dims]

elif EXTRA == "_sqrt_large":
    dims = [1, 2, 4, 25, 50, 100, 250, 500, 1000, 2000]
    ns = [int(50 * d**0.5) for d in dims]

elif EXTRA == "_sqrt_ld":
    dims = [1, 2, 4, 25, 50, 100]
    ns = [2 * int(d**0.5) for d in dims]

elif EXTRA == "_linear_ld":
    dims = [1, 2, 4, 25, 50, 100]
    ns = [2 * int(d) for d in dims]

elif EXTRA == "_quad_ld":
    dims = [1, 2, 4, 25, 50, 100]
    ns = [2 * int(d**1.2) for d in dims]

elif EXTRA == "_cub_ld":
    dims = [100] # [1, 2, 4, 25, 50, 100]
    ns = [2 * int(d**2.2) for d in dims]

elif EXTRA == "_gamma":
    dims = [50]
    ns = [50] * len(dims)

elif EXTRA == "_gammaksd":
    dims = [27]
    ns = [50] * len(dims)

elif EXTRA == "_gammammd":
    dims = [27]
    ns = [20] * len(dims)

if __name__ == "__main__":
    # D_n values
    print("dims:", dims)
    print("n:", ns)
    STATS_VALS_DIR = f"{args.DIR}/stats_res_rep_{SUFFIX}.p"
    statistic_res_list = hd.compute_statistic_rep(
        nexperiments=N_EXP,
        ns=ns,
        dims=dims,
        nreps=100,
        kernel_class=KERNEL,
        bandwidth_order=BANDWIDTH_POWER,
        delta=DELTA,
        statistic=STAT,
        print_once_every=max(N_EXP//10, 1),
        bandwidth_scale=GAM_SCALE,
    )

    # save empirical results
    pickle.dump(
        statistic_res_list,
        open(STATS_VALS_DIR, "wb")
    )

    print("Results saved to:", STATS_VALS_DIR)
