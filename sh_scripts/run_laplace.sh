method=all

hist_file="./res/laplace/_hist.txt"
echo "$(date +"%T") run history" > $hist_file

# # dim
for dim in 1 10 20 30 40 50 60 70 80 90 100
do
    CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 \
    experiments.py --model=laplace --dim=$dim --T=10 --n=1000 --nrep=100 --rand_start=10. --method=$method &
done
wait
echo "$(date +"%T") finished dim" >> $hist_file
