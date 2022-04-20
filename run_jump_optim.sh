# CUDA_VISIBLE_DEVICES=6 python -m cProfile -o prof_out \
# CUDA_VISIBLE_DEVICES=6 python \
#   experiments/ksd_jump.py --model=bimodal --k=1 --dim=5 --T=2 --n=1000 --ratio_t=0.5 --ratio_s=1. --load=res/bootstrap

# for dim in 60 70 80 90 #1 2 5 10 15 20 25 30 35 40 45 50 100
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 1-10 python \
#     experiments/ksd_jump.py --model=bimodal --k=1 --dim=$dim --T=10 --n=1000 --ratio_t=0.3 --ratio_s=.7
# done

for ratio_s in 0. 0.2 0.4 0.6 0.8 1.0 # 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.
do
    CUDA_VISIBLE_DEVICES="" taskset -c 1-30 python \
    experiments/ksd_jump.py --model=bimodal --k=1 --dim=1 --T=10 --n=1000 --ratio_t=0.5 --ratio_s=$ratio_s --nrep=1000 &
done
wait
for ratio_s in 0.1 0.3 0.5 0.7 0.9
do
    CUDA_VISIBLE_DEVICES="" taskset -c 1-30 python \
    experiments/ksd_jump.py --model=bimodal --k=1 --dim=1 --T=10 --n=1000 --ratio_t=0.5 --ratio_s=$ratio_s --nrep=1000 &
done

# CUDA_VISIBLE_DEVICES=6 python experiments/ksd_jump.py --model=gaussianmix --nmodes=4 --T=50 --n=1000 --seed=0
# CUDA_VISIBLE_DEVICES=6 python experiments/ksd_jump.py --model=gaussianmix --nmodes=4 --T=50 --n=1000 --seed=1
# CUDA_VISIBLE_DEVICES=6 python experiments/ksd_jump.py --model=gaussianmix --nmodes=4 --T=50 --n=1000 --seed=2
# CUDA_VISIBLE_DEVICES=7 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000 --seed=0
# CUDA_VISIBLE_DEVICES=7 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000 --seed=1
# CUDA_VISIBLE_DEVICES=7 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000 --seed=2
# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000
# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=gaussianmix --nmodes=10 --T=50 --n=1000

# CUDA_VISIBLE_DEVICES=6 python experiments/ksd_jump.py --model=gauss-scaled --T=50 --n=1000 --ratio_s=0.3
# CUDA_VISIBLE_DEVICES=7 python experiments/ksd_jump.py --model=gauss-scaled --T=50 --n=1000 --ratio_s=0.3 --seed=0
# CUDA_VISIBLE_DEVICES=7 python experiments/ksd_jump.py --model=gauss-scaled --T=50 --n=1000 --ratio_s=0.3 --seed=1

# CUDA_VISIBLE_DEVICES=6 python experiments/ksd_jump.py --model=gaussianmix --nmodes=4 --T=50 --n=1000 --seed=0 --load=res/bootstrap
# CUDA_VISIBLE_DEVICES=6 python experiments/ksd_jump.py --model=gaussianmix --nmodes=4 --T=50 --n=1000 --seed=1 --load=res/bootstrap
# CUDA_VISIBLE_DEVICES=6 python experiments/ksd_jump.py --model=gaussianmix --nmodes=4 --T=50 --n=1000 --seed=2 --load=res/bootstrap

# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000 --seed=0 --load=res/bootstrap
# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000 --seed=1 --load=res/bootstrap
# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000 --seed=2 --load=res/bootstrap

# CUDA_VISIBLE_DEVICES=7 python experiments/ksd_jump.py --model=t-banana --nmodes=10 --nbanana=5 --T=50 --n=1000

# CUDA_VISIBLE_DEVICES=6 python experiments/ksd_jump.py --model=rbm --dim=5 --dh=3 --shift=6. --T=50 --n=1000
# CUDA_VISIBLE_DEVICES=3 python experiments/ksd_jump.py --model=rbm --dim=10 --dh=3 --shift=6. --T=50 --n=1000
# CUDA_VISIBLE_DEVICES=2 python experiments/ksd_jump.py --model=rbm --dim=20 --dh=3 --shift=6. --T=50 --n=1000
# CUDA_VISIBLE_DEVICES=1 python experiments/ksd_jump.py --model=rbm --dim=30 --dh=3 --shift=6. --T=50 --n=1000
# CUDA_VISIBLE_DEVICES=0 python experiments/ksd_jump.py --model=rbm --dim=50 --dh=3 --shift=6. --T=50 --n=1000


## barker
# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=bimodal --k=2 --T=50 --n=1000 \
#   --ratio_t=0.3 --ratio_s=0.7 --seed=0 --load=res/bootstrap --mcmckernel=barker
# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=bimodal --k=2 --T=50 --n=1000 \
#   --ratio_t=0.5 --ratio_s=0.7 --seed=0 --load=res/bootstrap --mcmckernel=barker
# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=bimodal --k=2 --T=50 --n=1000 \
#   --ratio_t=0.7 --ratio_s=0.3 --seed=0 --load=res/bootstrap --mcmckernel=barker

# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=bimodal --k=2 --T=50 --n=1000 \
#   --ratio_t=0.7 --ratio_s=0.3 --seed=1 --load=res/bootstrap --mcmckernel=barker

# CUDA_VISIBLE_DEVICES=7 python experiments/ksd_jump.py --model=gauss-scaled --T=50 --n=1000 --ratio_s=0.3 --seed=0 --mcmckernel=barker
# CUDA_VISIBLE_DEVICES=7 python experiments/ksd_jump.py --model=gauss-scaled --T=50 --n=1000 --ratio_s=0.3 --seed=1 --mcmckernel=barker

# CUDA_VISIBLE_DEVICES=4 python experiments/ksd_jump.py --model=gaussianmix --nmodes=4 --T=50 --n=1000 --seed=0 --mcmckernel=barker
# CUDA_VISIBLE_DEVICES=4 python experiments/ksd_jump.py --model=gaussianmix --nmodes=4 --T=50 --n=1000 --seed=1 --mcmckernel=barker
# CUDA_VISIBLE_DEVICES=4 python experiments/ksd_jump.py --model=gaussianmix --nmodes=4 --T=50 --n=1000 --seed=2 --mcmckernel=barker

# CUDA_VISIBLE_DEVICES=3 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000 --seed=0 --mcmckernel=barker
# CUDA_VISIBLE_DEVICES=3 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000 --seed=1 --mcmckernel=barker
# CUDA_VISIBLE_DEVICES=3 python experiments/ksd_jump.py --model=gaussianmix --nmodes=5 --T=50 --n=1000 --seed=2 --mcmckernel=barker

# CUDA_VISIBLE_DEVICES=5 python experiments/ksd_jump.py --model=t-banana --nmodes=4 --nbanana=2 --T=50 --n=1000 --mcmckernel=barker