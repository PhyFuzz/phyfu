<div align="center">

<img src="logo.svg" width=300>

# PhyFu: A Fuzzer for Modern Physics Simulation Engines

<a href="#"> ![Version](https://img.shields.io/badge/PhyFu%200.1.0-brightgreen?style=for-the-badge) </a>
<a href="#"> ![Errors](https://img.shields.io/badge/%23errors-5K%2B-9cf?style=for-the-badge) </a>
<a href="#"> ![Crashes](https://img.shields.io/badge/%23crashes-20%2B-7289da?style=for-the-badge) </a>
<a href="#"> ![License](https://img.shields.io/badge/license-GitHub-green?style=for-the-badge)</a>

A Logic Bug Detector for Physics Simulation Engines

[Summary](#summary)
•
[Installation](#-installation)
•
[Quick Start](#-quick-start)
<br>
[Philosophy](#-philosophy)
•
[Credits](#credits)

</div>

<div align="center">

<br>

## Summary

</div>

PhyFu (_Phy_ - physics, _Fu_ - fuzzer) is a fuzzing framework designed specifically for Physics Simulation Engines (PSEs) to uncover logic errors. PhyFu can holistically detect errors in both the forward and backward simulation phase of PSEs.

### What is PhyFu?

PhyFu is a fuzzing framework for uncovering *logic errors*, i.e., errors that silently cause incorrect results without directly causing crashes, in the Physics Simulation Engines (or abbreviated as PSEs). To date, PhyFu has detected over 5K error-triggering inputs and over 20 crashes in 8 combinations of PSEs and physical scenarios.

To learn more about the philosophy of the project check the [philosophy](#-philosophy) section.

###### :exclamation: **IMPORTANT**: PhyFu is young software and just experienced a major refactoring process. Sometimes the code may not work as expected.

## 🔧 Installation

**PhyFu requires a Linux machine with CUDA drivers and cudatoolkit to operate.**

You can install it through your favorite environment manager:

-
  <details>
  <summary><a href="https://docs.conda.io/en/latest/">conda</a></summary>

  ```bash
  git clone https://github.com/PhyFuzz/phyfu.git
  cd phyfu
  conda create -n phyfu python=3.8
  conda activate phyfu
  make develop
  ```

  </details>

- <details>

  ```bash
  git clone https://github.com/PhyFuzz/phyfu.git
  cd phyfu
  python3 -m pip install --user --upgrade pip
  python3 -m pip install --user virtualenv
  python3 -m venv env
  source env/bin/activate
  make develop
  ```

  <summary><a href="https://virtualenv.pypa.io/en/latest/">virtualenv</a></summary>
  </details>

The last step of the installation process may take several to dozens of minutes, depending on your network speed.

So far, PhyFu is installed in development mode, so that you can easily modify the code without need to rerun the installation process.
To install PhyFu in production mode, run `make install` instead of `make develop`. This will install PhyFu in your system's Python environment.

## ⚙ Quick Start

To quickly try our tool, run the following to fuzz the `Taichi` PSE with 200 seeds and seed scheduling enabled:

```bash
phyfu.fuzz taichi two_balls both --test_times 200 --seed_getter art
```

Wait for about 3 minutes, and you will see the following output in the end:

```txt
#loss_too_large: 17
#deviated_init_state: 3
```

It means that PhyFu has found 17 errors that are caused by the loss value being too large (backward errors), and 3 errors that are caused by the deviation of the initial state (forward errors).

#### It works, cool! What are the next steps?

There are three things that you can do next:

1. Run large scale experiments to evaluate PhyFu's effectiveness.
2. Analyze the errors found by PhyFu in the large scale experiments.
3. Extend PhyFu to test more PSEs and physical scenarios.

Setting these up is discussed in the [readthedocs](https://phyfu.readthedocs.io/en/latest/), so be sure to check there if you are interested!

## ❓ Philosophy

One of the key challenges in detecting logic errors is designing an oracle that can decide whether the output results are wrong or not. Based on principled physics laws, PhyFu proposes two testing oracles that can holistically detect errors in both the forward and backward simulation phases of PSEs. Specifically,

1. PhyFu mutates initial states and asserts if the PSE under test behaves consistently with respect the two testing oracles.
2. Further with feedback-driven test case scheduling, the search process for errors is significantly accelerated.
3. PhyFu also leverages a set of design choices to ensure the validity of testing seeds.

## 📚 FAQ

The [readthedocs](https://phyfu.readthedocs.io/en/latest/) is the go-to place if you need answers to anything PhyFu-related. Usage, APIs, Extensions?
It's all there, so we recommend you seriously go [read it](https://phyfu.readthedocs.io/en/latest/)!

## 🍀 Misc

The hyper-parameters we use are all listed in `phyfu/configs/`. For the meaning and usage of each hyper-parameter, please refer to the [readthedocs](https://phyfu.readthedocs.io/en/latest/extension.html).

For the omitted proofs of some theorems in the paper, please refer to [here](https://github.com/PhyFuzz/phyfu/tree/main/docs/proof.pdf).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.
This REAME file is adapted from [`nvim-neorg/neorg`](https://github.com/nvim-neorg/neorg).
