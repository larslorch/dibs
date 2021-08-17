# DiBS: Differentiable Bayesian Structure Learning

This is the Python implementation for *DiBS*  ([Lorch et al., 2021](https://arxiv.org/abs/2105.11839)), a fully differentiable method for joint Bayesian inference of graphs and parameters of general Bayesian networks.

The repository consists of two branches:

- `master` (recommended): Lightweight and easy-to-use code for using DiBS in your research or applications.
- `full`: Comprehensive code to reproduce the experimental results in ([Lorch et al., 2021](https://arxiv.org/abs/2105.11839)). The purpose of this branch is reproducibility; the branch is not updated anymore and may contain outdated notation and documentation.

In this work, DiBS is instantiated with the general-purpose particle variational inference method *SVGD*  ([Liu and Wang, 2016](https://arxiv.org/abs/1608.04471)). The implementation uses [JAX](https://github.com/google/jax). The end-to-end nature of DiBS allows for just-in-time compilation and automatic differentiation.

## Quick start

DiBS translates inferring posteriors over Bayesian networks into inference over continuous latent variables.
For a working example of inferring the joint posterior over Gaussian Bayes net graphs and parameters, we recommend opening our example notebook in Google colab, which runs **directly from your browser**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/larslorch/dibs/blob/master/examples/dibs_joint_colab.ipynb)

The analogous [jupyter notebook](examples/dibs_joint.ipynb) can be found inside the `examples/` folder.

Executing the notebook as is will perform posterior inference over Bayes net graph and parameters using DiBS with SVGD and generate a visualization of the matrices of edge probabilities modeled by the individual transported particles.

<br/><br/>
<p align="center">
	<img src="./examples/dibs_joint.gif" width="90%">
</p>
<br/><br/>

## Installation

The provided code is run with `python`. We recommend using `conda`. First clone the code repository:
```
git clone https://github.com/larslorch/dibs.git
```


If you want to set up a new `conda` environment, you can run the following commands:
```
cd dibs
conda env create --file environment.yml
conda activate dibs
pip install -e .
```
Note the "`.`" in the last call. 
If you want to use DiBS within an existing `conda` environment or `virtualenv`, you can just run 
```
pip install -e .
```
to set up the `dibs` package. 


## Reference
   
If you find this code useful, please cite our paper: 

```
@article{lorch2021dibs,
  title={DiBS: Differentiable Bayesian Structure Learning},
  author={Lorch, Lars and Rothfuss, Jonas and Sch{\"o}lkopf, Bernhard and Krause, Andreas},
  journal={arXiv preprint arXiv:2105.11839},
  year={2021}
}
```
