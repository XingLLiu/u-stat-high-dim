## 1. intro figure
## i. ksd fixed n
DIR_DIST=res/check
mkdir $DIR_DIST

N_EXP=30
R=-1. # use med-heuristic

CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 src/high_dim/stats.py \
    --R=$R --STAT=ksd --KERNEL=RBF --N_EXP=$N_EXP --DIR=$DIR_DIST &

CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 src/high_dim/moments.py \
    --R=$R --STAT=ksd --KERNEL=RBF --NPOP=4000 --DIR=$DIR_DIST 

# ii. Linear-MMD fixed n
CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 src/high_dim/stats.py \
    --R=$R --STAT=mmd --KERNEL=Linear --DELTA=10. --N_EXP=$N_EXP --DIR=$DIR_DIST &

CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 src/high_dim/moments.py \
    --R=$R --DELTA=10. --R=1. --STAT=mmd --KERNEL=Linear --NPOP=10000 --DIR=$DIR_DIST


# # 2. MMD RBF 
# DIR_DIST=res/check
# mkdir $DIR_DIST

# N_EXP=30
# R=-1. # use med-heuristic

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 src/high_dim/stats.py \
#     --R=$R --STAT=mmd --KERNEL=RBF --N_EXP=$N_EXP --DIR=$DIR_DIST &

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 src/high_dim/stats.py \
#     --R=$R --STAT=mmd --KERNEL=RBF --N_EXP=$N_EXP --DIR=$DIR_DIST --EXTRA=ld &

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 src/high_dim/moments.py \
#     --R=$R --STAT=mmd --KERNEL=RBF --NPOP=-1 --DIR=$DIR_DIST

# # 3. ksd distances
# DIR_DIST=res/check_dist
# mkdir $DIR_DIST

# N_EXP=30
# R=-1. # use med-heuristic

# mkdir $DIR_DIST
# CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 src/high_dim/stats.py \
#     --R=$R --STAT=ksd --KERNEL=RBF --EXTRA=quad --N_EXP=$N_EXP --DIR=$DIR_DIST &

# CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 src/high_dim/stats.py \
#     --R=$R --STAT=ksd --KERNEL=RBF --N_EXP=$N_EXP --DIR=$DIR_DIST &

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 src/high_dim/stats.py \
#     --R=$R --STAT=ksd --KERNEL=RBF --EXTRA=sqrt --N_EXP=$N_EXP --DIR=$DIR_DIST &


# # 4. KSD varying gamma
# DIR_GAMMA=res/gamma_dim27
# EXTRA=gammaksd
# mkdir $DIR_GAMMA

# N_EXP=30

# for gam_scale in 3.5 4. 5. 6. 7. 8. 10. 16. 20.
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 src/high_dim/moments.py \
#         --R=0. --GAM_SCALE=$gam_scale --STAT=ksd --KERNEL=RBF --EXTRA=$EXTRA --DIR=$DIR_GAMMA --NPOP=10000 #&

#     CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 src/high_dim/stats.py \
#         --R=0. --GAM_SCALE=$gam_scale --STAT=ksd --KERNEL=RBF --EXTRA=$EXTRA --DIR=$DIR_GAMMA

#     mv $DIR_GAMMA/res_analytical_delta2.0_r0.0_ksd_RBF_$EXTRA.csv $DIR_GAMMA/res_analytical_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.csv
#     mv $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_ksd_RBF_$EXTRA.p $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.p
# done

wait