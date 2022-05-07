## training
CUDA_VISIBLE_DEVICES=2,6,7 python src/nf_maf.py


# CUDA_VISIBLE_DEVICES="" taskset -c 21-35 python \
#   experiments/ksd_all.py --model=nf --T=100 --n=100 --nrep=50 --mcmckernel=barker --threshold=3e10 &

# CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python \
#   experiments/ksd_all.py --model=nf --T=100 --n=50 --nrep=50 --mcmckernel=barker --threshold=3e10
