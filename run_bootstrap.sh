# CUDA_VISIBLE_DEVICES=0,4 python experiments/bootstrap_test_convolved.py
# CUDA_VISIBLE_DEVICES=0,4 python experiments/bootstrap_test.py
CUDA_VISIBLE_DEVICES=5 python experiments/bootstrap_test_convolved_multiple_kdim.py --load=res/bootstrap --ratio_t=0.5 --ratio_s=1.
CUDA_VISIBLE_DEVICES=5 python experiments/bootstrap_test_convolved_multiple_kdim.py --load=res/bootstrap --ratio_t=0.3 --ratio_s=7.