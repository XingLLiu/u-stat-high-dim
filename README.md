# Code for A High-dimensional Convergence Theorem for U-statistics with Applications to Kernel-based Testing
This repo contains the code for the experiments in the paper:

> [Huang, Kevin H. and Liu, Xing and Duncan, Andrew B. and Gandy, Axel (2023). A High-dimensional Convergence Theorem for U-statistics with Applications to Kernel-based Testing. arXiv preprint arXiv: 2302.05686](https://arxiv.org/abs/2302.05686)

## How to install?
Packages that this programme depends on are listed in `setup.py`. Before running any scripts, run the following to install the current package and the dependencies
```bash
pip install -e .
```

## Examples
To reproduce results in the paper, run e.g.,
```bash
# generate results for the intro plot
sh run.sh
```
Results will be stored in `res/`. Plots can then be made using the jupyter notebooks in `figs_code`. E.g., the intro plot Figure 1 can be made using `figs_code/intro.ipynb`.

## Folder structure

```bash
.
├── figs                          # Folder for storing figures
├── figs_code                     # Jupyter notebooks for making plots
├── res                           # Folder for storing results
├── src                           # Source files for KSD, MMD and kernels
├── run.sh                        # Shell script for generating results
├── setup.py                      # Setup file for easy-install
└── README.md
```

