CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python \
  experiments/ksd_all.py --model=nf --T=50 --n=1000 --nrep=100 --method=mcmc_all
