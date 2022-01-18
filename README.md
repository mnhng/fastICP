Code for the following paper:
> [Efficient Identification of Direct Causal Parents via Invariance and Minimum Error Testing](https://arxiv.org/abs/2409.12797)

### Requirements

To setup the required environment

1. Install [*miniconda*](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
2. Run `conda env create -n icp -f env.yml`

### Examples

- Run the example of a diamond graph using `run_toy.py`
- The simulation examples are in the `scripts` folder.

#### Kemmeren experiment

1. Find parents using ICP, MMSE-ICP, and fastICP by running `scripts/kemmeren.sh` (**Warning**: this may take some time)
2. Plot the result by running `plot_kemmeren.py`

### Citation

If you find this code useful, please cite:

```
@article{nguyen2024fasticp,
    title={{Efficient Identification of Direct Causal Parents via Invariance and Minimum Error Testing}},
    author={Nguyen, Minh and Sabuncu, Mert R.},
    journal={Transactions on Machine Learning Research (TMLR)},
    year={2024}
}
```
