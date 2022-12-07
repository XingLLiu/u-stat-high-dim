# 1h30min at 2%
# new: 1h25min at 2%
# method=spksd # 48mins at 4%
# method=pksd
# method=all # 2h2min at 1%
# CUDA_VISIBLE_DEVICES="" taskset -c 1-50 python3  \
#     experiments.py --model=rbm --dim=50 --dh=10 --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method


hist_file="./res/rbm/_hist.txt"
echo "$(date +"%T") run history" > $hist_file

method=all

# latent dim
for dh in 2 5 8 10 # 20 30 40
do
  CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
    experiments.py --model=rbm --dim=50 --dh=$dh --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
done
wait
echo "$(date +"%T") finished latent dim 1/2" >> $hist_file
for dh in 20 30 40
do
  CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
    experiments.py --model=rbm --dim=50 --dh=$dh --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
done
wait
echo "$(date +"%T") finished latent dim 2/2" >> $hist_file

# # perturbation to mixing ratios
# #! some of these did not finish because log_prob and log_prob_np yield different values
for shift in 0. 0.01 0.05 #0.1 0.5 1.
do
  CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
    experiments.py --model=rbm --dim=50 --dh=5 --shift=$shift --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
done
wait
echo "$(date +"%T") finished mixing ratios 1/2" >> $hist_file
for shift in 0.1 0.5 1.
do
  CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
    experiments.py --model=rbm --dim=50 --dh=5 --shift=$shift --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
done
wait
echo "$(date +"%T") finished mixing ratios 2/2" >> $hist_file

# # #! some of these did not finish because log_prob and log_prob_np yield different values
for n in 200 500 1000 #1500 2000
do
  # power test
  CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 \
    experiments.py --model=rbm --dim=50 --dh=5 --shift=1. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
  
  # level test
  CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 \
    experiments.py --model=rbm --dim=50 --dh=5 --shift=0. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
done
wait
echo "$(date +"%T") finished sample size 1/2" >> $hist_file
for n in 1500 2000
do
  # power test
  CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 \
    experiments.py --model=rbm --dim=50 --dh=5 --shift=1. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
  
  # level test
  CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 \
    experiments.py --model=rbm --dim=50 --dh=5 --shift=0. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
done
wait
echo "$(date +"%T") finished sample size 2/2" >> $hist_file
