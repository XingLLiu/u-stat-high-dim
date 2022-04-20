# ratio_s
for ratio_s in 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.
do
    CUDA_VISIBLE_DEVICES="" taskset -c 31-45 python \
    experiments/ksd_all.py --model=bimodal --k=1 --dim=50 --T=10 --n=1000 --ratio_t=0.5 --ratio_s=$ratio_s --delta=6 \
    --mcmckernel=barker --nrep=100 --suffix=_single &
done
# wait

# n power test
for n in 200 500 1000 1500 2000
do
    CUDA_VISIBLE_DEVICES="" taskset -c 31-45 python \
    experiments/ksd_all.py --model=bimodal --k=1 --dim=50 --T=10 --n=$n --ratio_t=0.5 --ratio_s=0.3 --delta=6 \
    --mcmckernel=barker --nrep=100 --suffix=_single &
done
wait

# # n level test
# for n in 200 500 1000 1500 2000
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python \
#     experiments/ksd_all.py --model=bimodal --k=1 --dim=50 --T=10 --n=$n --ratio_t=0.5 --ratio_s=0.5 --delta=6 \
#     --mcmckernel=barker --nrep=100 &
# done
# wait