## ksd RBF
# varying gamma
DIR_GAMMA=res/high_dim/gamma_ksd
EXTRA=gammaksd
STAT=ksd
K=RBF

mkdir $DIR_GAMMA
for gam_scale in 0.01 0.05 0.1 0.5 1. 5. 10.
do
    # estimate moments
    CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 high_dim/moments.py \
        --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=$STAT --KERNEL=$K --EXTRA=$EXTRA --DIR=$DIR_GAMMA --NPOP=10000

    # compute Dn
    CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
        --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=$STAT --KERNEL=$K --EXTRA=$EXTRA --DIR=$DIR_GAMMA

    # rename files
    mv $DIR_GAMMA/res_analytical_delta2.0_r0.0_ksd_RBF_$EXTRA.csv $DIR_GAMMA/res_analytical_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.csv
    mv $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_ksd_RBF_$EXTRA.p $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.p
done
