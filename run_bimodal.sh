# # ratio_s
# for ratio_s in 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 26-35 python \
#     experiments/ksd_all.py --model=bimodal --k=1 --dim=1 --T=10 --n=1000 --ratio_t=0.5 --ratio_s=$ratio_s --delta=6 \
#     --mcmckernel=barker --nrep=100 --rand_start=10. &
# done

# # dim
# for dim in 1 10 20 30 40 50 60 70 80 90 100
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python \
#     experiments/ksd_all.py --model=bimodal --k=1 --dim=$dim --T=10 --n=1000 --ratio_t=0.5 --ratio_s=1. --delta=6 \
#     --mcmckernel=barker --nrep=100 --rand_start=10. &

#     CUDA_VISIBLE_DEVICES="" taskset -c 11-21 python \
#     experiments/ksd_all.py --model=bimodal --k=1 --dim=$dim --T=10 --n=1000 --ratio_t=0.5 --ratio_s=.3 --delta=6 \
#     --mcmckernel=barker --nrep=100 --rand_start=10. &
# done

# # n power test
for n in 200 500 1000 1500 2000
do
    CUDA_VISIBLE_DEVICES="" taskset -c 0-15 python \
    experiments/ksd_all.py --model=bimodal --k=1 --dim=50 --T=10 --n=$n --ratio_t=0.5 --ratio_s=0.3 --delta=6 \
    --mcmckernel=barker --nrep=100 --rand_start=10. &
done
wait
# # n level test
for n in 200 500 1000 1500 2000
do
    CUDA_VISIBLE_DEVICES="" taskset -c 16-30 python \
    experiments/ksd_all.py --model=bimodal --k=1 --dim=50 --T=10 --n=$n --ratio_t=0.5 --ratio_s=0.5 --delta=6 \
    --mcmckernel=barker --nrep=100 --rand_start=10. &
done

# inter-modal distance
# redo fig1 and 2 in paper
# for delta in 1 2 3 4 5 6 7 8 9 10 11 12
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python \
#     experiments/ksd_all.py --model=bimodal --k=1 --dim=1 --T=20 --n=1000 --ratio_t=0.5 --ratio_s=1. --delta=$delta \
#     --mcmckernel=barker --nrep=100 --rand_start=10. &
# done

# CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python \
#     experiments/ksd_all.py --model=bimodal --k=1 --dim=1 --T=10 --n=1000 --ratio_t=0.5 --ratio_s=1. --delta=3 \
#     --mcmckernel=barker --nrep=100 --seed=1234 &
wait