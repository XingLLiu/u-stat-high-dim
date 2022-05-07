import src.ksd.models as models
from src.ksd.find_modes import find_modes, pairwise_directions
from src.ksd.langevin import RandomWalkMH, RandomWalkBarker
import src.ksd.langevin as mcmc
from src.ksd.ksd import KSD
from src.ksd.kernel import IMQ, l2norm
from src.ksd.bootstrap import Bootstrap
from src.ksd.find_modes import find_modes, pairwise_directions
from tqdm import tqdm

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import time

from src.sensors import Sensor, SensorImproper

MCMCKernel = RandomWalkBarker # RandomWalkBarker 

MODEL = "modified"
T_LIST = [10, 50] # [1000, 2000, 4000, 6000, 8000, 10000, 20000]

if MODEL == "modified":
    # Observation indicators from the fifth sensor (1st column) to the first four sensors
    # and those from the sixth sensor (2nd column) to the first four sensors.
    Ob = tf.constant([1., 0, 1, 0, 1, 0, 1, 0])
    Ob = tf.transpose(tf.reshape(Ob, (2, -1)))

    # Observation indicators among the first four sensors. 
    Os = tf.constant([[0., 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [1, 1, 0, 0]])

    # Each row indicates the location of the known sensors (5th and 6th).
    Xb = tf.constant([0.5, 0.3, 0.3, 0.7])
    Xb = tf.transpose(tf.reshape(Xb, (2, -1)))

    # Each row indicates the location of the unknown sensors (1st, 2nd, 3rd, and 4th).
    Xs = tf.constant([0.5748, 0.0991, 0.2578, 0.8546, 
                0.9069, 0.3651, 0.1350, 0.0392])
    Xs = tf.transpose(tf.reshape(Xs, (2, -1)))

    # The observed distances from the fifth sensor (1st column) to the first four sensors
    # and those from the sixth sensor (2nd column) to the first four sensors.
    Yb = tf.constant([0.6103, 0, 0.2995, 0, 
                0.3631, 0, 0.5656, 0])
    Yb = tf.transpose(tf.reshape(Yb, (2, -1)))

    # Observed distances among the first four sensors.
    Ys = tf.constant([[0, 0, 0, 0.9266],
                [0, 0, 0.2970, 0.8524],
                [0, 0.2970, 0, 0],
                [0.9266, 0.8524, 0, 0]])


    loc_true = tf.constant([
        0.57, 0.91,
        0.10, 0.37,
        0.26, 0.14,
        0.85, 0.04,
    #     0.50, 0.30,
    #     0.30, 0.70
    ])

    ModelClass = Sensor
    path = "res/sensors/res_ram.csv"
    prefix = ""

elif MODEL == "original":
    loc_true = tf.constant([
        [0.125, 0.81],
        [0.225, 0.475],
        [0.35, 0.1],
        [0.45, 0.22],
        [0.55, 0.73],
        [0.57, 0.93],
        [0.85, 0.05],
        [0.85, 0.8],
        [0.3, 0.7],
        [0.5, 0.3], 
        [0.7, 0.7],])

    dist_true = tf.math.sqrt(l2norm(loc_true, loc_true))
    dist_noise = tf.random.normal((loc_true.shape[0], loc_true.shape[0])) * 0.02
    assert dist_noise.shape == (11, 11)
    dist_noise = tf.experimental.numpy.triu(dist_noise, 1)
    dist_noise += tf.transpose(dist_noise)
    dist_true += dist_noise
    assert tf.experimental.numpy.allclose(dist_true, tf.transpose(dist_true))

    # Observation indicators for the sensors
    ## potentially wrong links (1 -- 11): 
    ## (3, 8)
    O_true = tf.constant([
        [0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
        [0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.],
        [0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.],
        [1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
    ])
    assert tf.experimental.numpy.allclose(O_true, tf.transpose(O_true))

    # Observation indicators from the fifth sensor (1st column) to the first four sensors
    # and those from the sixth sensor (2nd column) to the first four sensors.
    Ob = O_true[:8, 8:] # 8 x 3

    # Observation indicators among the first four sensors. 
    Os = O_true[:8, :8] # 8 x 8

    # Each row indicates the location of the known sensors (9th to 11th).
    Xb = loc_true[8:]

    # Each row indicates the location of the unknown sensors (1st to 8th).
    Xs = loc_true[:8]

    # The observed distances from the observed sensors (each col) to the unobserved sensors.
    Yb = (dist_true * O_true)[:8, 8:] # 8 x 3

    # Observed distances among the first 8 sensors.
    Ys = (dist_true * O_true)[:8, :8] # 8 x 8

    ModelClass = SensorImproper
    path = "res/sensors/res_ram_improper.csv"
    prefix = "improper_"

tf.random.set_seed(1)

target = ModelClass(Ob, Os, Xb, Xs, Yb, Ys)
log_prob_fn = target.log_prob

dim = Ys.shape[0] * 2
print("model class:", MODEL, "; dim:", dim)
print("loading data from", path)

## load data
n = 2000
# ind = tf.range(start=0, limit=200000, delta=200000//n)
ind = tf.range(start=0, limit=100000, delta=100000//n)

mcmc_res = pd.read_csv(path)
# mcmc_res = pd.read_csv("res/sensors/res_mt.csv")
samples_off = tf.constant(mcmc_res.loc[ind].to_numpy(), dtype=tf.float32)

## split to train and test
tf.random.set_seed(1)

ntrain = n // 2

samples_init = samples_off

samples_init = tf.random.shuffle(samples_init) # shuffle
# samples_init = samples_init[::-1] #!

sample_off_train, sample_off_test = samples_init[:ntrain, ], samples_init[ntrain:, ]


def experiment(T_list, n, target_dist, sample_init_train, sample_init_test):
    Tmax = T_list[-1]
    jump_ls = tf.linspace(0.8, 1.2, 51)
    
    ntrain = n // 2
    threshold = 1e-4
    nrep = 1

    num_boot = 800
    alpha = 0.05
    
    kernel = IMQ(med_heuristic=True)
    ksd = KSD(target=target_dist, kernel=kernel)
    bootstrap = Bootstrap(ksd, n-ntrain)
    
    multinom_samples = bootstrap.multinom.sample((nrep, num_boot))

    p_val_list = []
    unscaled_ksd_list = []
    jump_ratio_list = []

    i = 0
    # sample data
    start_pts = tf.concat([
        sample_init_train[:(ntrain//2)], 
        tf.random.uniform(shape=(ntrain//2, dim), minval=0., maxval=1.)], axis=0)
    
    # find modes
    tic = time.perf_counter()
    mode_list, inv_hess_list = find_modes(start_pts, log_prob_fn, grad_log=None, threshold=threshold)
    toc = time.perf_counter()
    print(f"Optimisation finished in {toc - tic:0.4f} seconds")
    print(f"Num. modes found: {len(mode_list)}")

    # #!
    # inv_hess_list = [tf.eye(8)] * len(inv_hess_list)

    proposal_dict = mcmc.prepare_proposal_input_all(mode_list=mode_list, inv_hess_list=inv_hess_list)
    _, ind_pair_list = pairwise_directions(mode_list, return_index=True)

    # find best jump scale
    print("running in parallel ...")
    tic = time.perf_counter()

    mh_jumps = MCMCKernel(log_prob=log_prob_fn)
    mh_jumps.run(steps=Tmax, std=jump_ls, x_init=sample_init_train, ind_pair_list=ind_pair_list, **proposal_dict)
    toc = time.perf_counter()
    print(f"... done in {toc - tic:0.4f} seconds")

    for T in tqdm(T_list):
        # compute ksd
        scaled_ksd_vals = []
        unscaled_ksd_temp = []
        for j in range(jump_ls.shape[0]):
            x_t = mh_jumps.x[j, T-1, :, :]
            _, ksd_val = ksd.h1_var(X=x_t, Y=tf.identity(x_t), return_scaled_ksd=True)
            
            scaled_ksd_vals.append(ksd_val.numpy())
        
        best_jump = jump_ls[tf.math.argmax(scaled_ksd_vals)]

        # mh perturbation
        mh = MCMCKernel(log_prob=log_prob_fn)
        mh.run(steps=T, std=best_jump, x_init=sample_init_test, 
            ind_pair_list=ind_pair_list, **proposal_dict)
        x_0 = mh.x[0, :, :]
        x_t = mh.x[-1, :, :]

        # compute p-value
        kernel = IMQ(med_heuristic=True)
        ksd = KSD(target=target_dist, kernel=kernel)
        bootstrap = Bootstrap(ksd, n)

        multinom_one_sample = multinom_samples[i, :]

        if T == T_list[0]:
            # no perturbation
            _, p_val0 = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=x_0, multinom_samples=multinom_one_sample)
            p_val_list.append(p_val0)
            unscaled_ksd_list.append(bootstrap.ksd_hat)
            jump_ratio_list.append(-1.)

        # after perturbation
        _, p_valt = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=x_t, multinom_samples=multinom_one_sample)        
        p_val_list.append(p_valt)
        unscaled_ksd_list.append(bootstrap.ksd_hat)
        jump_ratio_list.append(best_jump.numpy())
        
    res = pd.DataFrame({"T": [0] + T_list, "pval": p_val_list, "unscaled_ksd": unscaled_ksd_list, "jump": jump_ratio_list})
    res.to_csv(f"res/sensors/{prefix}hess_sensors_pvals_{T}.csv", index=False)
    
    return res

if __name__ == "__main__":
    tf.random.set_seed(1)
    tic = time.perf_counter()
    res_df = experiment(T_LIST, n, target,
                sample_init_train=sample_off_train,
                sample_init_test=sample_off_test,
                )
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.4f} seconds")
    
