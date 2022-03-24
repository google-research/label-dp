This repository contains code for the multi-stage label differential privacy
training code for the paper

> Badih Ghazi, Noah Golowich, Ravi Kumar, Pasin Manurangsi, Chiyuan Zhang. *Deep
> Learning with Label Differential Privacy*. Advances in Neural Information
> Processing Systems (**NeurIPS**), 2021.
> [arxiv:2102.06062](https://arxiv.org/abs/2102.06062)

**Note**: This is not an officially supported Google product.

## Abstract

The Randomized Response (`RR`) algorithm is a classical technique to improve
robustness in survey aggregation, and has been widely adopted in applications
with differential privacy guarantees. We propose a novel algorithm, *Randomized
Response with Prior* (`RRWithPrior`), which can provide more accurate results
while maintaining the same level of privacy guaranteed by `RR`. We then apply
`RRWithPrior` to learn neural networks with *label* differential privacy
(`LabelDP`), and show that when only the label needs to be protected, the model
performance can be significantly improved over the previous state-of-the-art
private baselines. Moreover, we study different ways to obtain priors, which
when used with `RRWithPrior` can additionally improve the model performance,
further reducing the accuracy gap between private and non-private models. We
complement the empirical results with theoretical analysis showing that
`LabelDP` is provably easier than protecting both the inputs and labels.

## Getting Started

### Requirements

This codebase is implemented with [Jax](https://github.com/google/jax),
[Flax](https://github.com/google/flax) and
[Optax](https://github.com/deepmind/optax). To install the dependencies, run the
following commands (see also the [Jax](https://github.com/google/jax) homepage
for the latest installation guides for GPU/TPU support).

```
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip install -r requirements.txt
```

### 2-stage LabelDP Training for CIFAR-10

Use the following command to run

```
python3 -m label_dp.main --base_workdir <result-dir> --profile_key <profile_key>
```

An example `<profile_key>` is `cifar10/e2/lp-2st/run0`. Check out
`profiles/p100_cifar10.py` for a list of all pre-defined profiles. One can also
run the following command to query a list of all registered profile keys
matching a given regex:

```
python3 -m label_dp.profiles <key_regex>
```

Note the original experiments in the paper were run with Tensorflow. This
codebase is a reimplementation with Jax. The results we obtained with this
codebase are slightly different from what was reported in the paper. We include
the numbers here for your reference (CIFAR-10 with 2-stage training):

Epsilon | Test Accuracy ± std | Accuracy from Table 1
------- | ------------------- | ---------------------
1.0     | 62.89 ± 2.07        | 63.67
2.0     | 88.11 ± 0.38        | 86.05
4.0     | 94.18 ± 0.13        | 93.37
8.0     | 95.18 ± 0.07        | 94.52

### Structure of the Code

To use the training code in the current setup, you just need to define new
profiles that specify hyperparameters such as what dataset to load, what
optimizer to use, etc. Create a new file `profiles/pXXX_xxx.py` and import it
from `profiles/__init__.py`. Functions starting with `register_` will be called
automatically to register hyperparameter profiles.

The code loads dataset using [tfds](https://www.tensorflow.org/datasets), so
theoretically any dataset available in tfds runs. But note for convenience we
load the entire dataset into memory when doing data splitting for multi-stage
training. This should work for CIFAR scale dataset.

The key algorithm `RRWithPrior` (Algorithm 2 from the paper) is implemented in
`rr_with_prior` (`train.py`) with numpy, and can be used independently in other
scenarios.

## Citation

```
@article{ghazi2021deep,
  title={Deep Learning with Label Differential Privacy},
  author={Ghazi, Badih and Golowich, Noah and Kumar, Ravi and Manurangsi, Pasin and Zhang, Chiyuan},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
