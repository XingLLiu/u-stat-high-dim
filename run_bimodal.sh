# for delta in 2. 4. 8. 12.
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 1-20 python \
#     experiments/bootstrap_test_mh_jump_optim.py --model=bimodal --k=1 --dim=1 --T=10 --n=1000 --ratio_t=0.3 --ratio_s=0.7 --delta=$delta --nrep=1000 &
# done

for n in 1000 1500 2000 2500 3000
do
    CUDA_VISIBLE_DEVICES="" taskset -c 10-30 python \
    experiments/bootstrap_test_mh_jump_optim.py --model=bimodal --k=1 --dim=10 --T=10 --n=$n --ratio_t=0.3 --ratio_s=0.7 --delta=8. --nrep=1000\
     --load=res/bimodal &
done