# # ratio_s
# for ratio_s in 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python \
#     experiments.py --model=bimodal --k=1 --dim=1 --T=10 --n=1000 --ratio_t=0.5 --ratio_s=$ratio_s --delta=6 \
#     --nrep=100 --rand_start=10. &
# done
# wait 

# # # dim
# for dim in 1 10 20 30 40 50 60 70 80 90 100
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python \
#     experiments.py --model=bimodal --k=1 --dim=$dim --T=10 --n=1000 --ratio_t=0.5 --ratio_s=1. --delta=6 \
#     --nrep=100 --rand_start=10. &
# done
# wait

# # n power test
# for n in 200 500 1000 1500 2000
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 0-5 python \
#     experiments.py --model=bimodal --k=1 --dim=50 --T=10 --n=$n --ratio_t=0.5 --ratio_s=1. --delta=6 \
#     --nrep=100 --rand_start=10. &
# done

# # n level test
# for n in 200 500 1000 1500 2000
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 6-10 python \
#     experiments.py --model=bimodal --k=1 --dim=50 --T=10 --n=$n --ratio_t=0.5 --ratio_s=0.5 --delta=6 \
#     --nrep=100 --rand_start=10. &
# done
# wait

# # inter-modal distance
for delta in 1 2 3 4 5 6 7 8 9 10 11 12 # 13 14 15
do
    CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python \
    experiments.py --model=bimodal --k=1 --dim=1 --T=10 --n=1000 --ratio_t=0.5 --ratio_s=1. --delta=$delta \
    --nrep=100 --rand_start=10. &
done
wait