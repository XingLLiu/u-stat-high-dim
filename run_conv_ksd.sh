# for ratio in 0.5 0.7
#     do
#         CUDA_VISIBLE_DEVICES=6 python experiments/compare_samplers_convolved.py --ratio=$ratio
#         CUDA_VISIBLE_DEVICES=6 python experiments/compare_samplers_convolved_med.py --ratio=$ratio
#     done
CUDA_VISIBLE_DEVICES=6 python experiments/bootstrap_test_mh_kdim.py --load=res/bootstrap --ratio_t=0.5 --ratio_s=1.
CUDA_VISIBLE_DEVICES=6 python experiments/bootstrap_test_mh_kdim.py --load=res/bootstrap --ratio_t=0.3 --ratio_s=7.