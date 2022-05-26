# Code for pKSD
See `example_gaussian_mix.ipynb` for an example of how to use pKSD tests for a given target distribution and a given sample.

To reproduce results in the paper, run e.g.,
```bash
# mixture of two gaussians example
sh sh_scripts/run_bimodal.sh
```
Results will be stored in `res/bimodal`. Other experiments can be reproduced similarly by changing `run_bimodal.sh` to `run_rbm.sh`, `run_t-banana.sh` and `run_sensors.sh`.

## Folder structure

```bash
.
├── src                           # Source files for pKSD and benchmarks
├── sh_scripts                    # Shell scripts to run experiments
├── res                           # Folder to store results
├── experiments.py                # Main script for generating results
├── example_gaussian_mix.ipynb    # Demonstration for how to use pKSD tests
├── setup.py                      # Setup file for easy-install of pKSD
└── README.md
```