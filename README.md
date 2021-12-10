# DiBS: Differentiable Bayesian Structure Learning

This is the Python JAX implementation for *DiBS*  ([Lorch et al., 2021](https://arxiv.org/abs/2105.11839)), a fully differentiable method for joint Bayesian inference of the DAG and parameters of general, causal Bayesian networks.

The repository consists of two branches:

- `master` (recommended): Lightweight and easy-to-use code for using DiBS in your research or applications.
- `full`: Comprehensive code to reproduce the experimental results in ([Lorch et al., 2021](https://arxiv.org/abs/2105.11839)). The purpose of this branch is reproducibility; the branch is not updated anymore and may contain outdated notation and documentation.

In this implementation, DiBS inference is performed with the particle variational inference method *SVGD*  ([Liu and Wang, 2016](https://arxiv.org/abs/1608.04471)). 
Since DiBS and SVGD operate on continuous tensors and solely rely on Monte Carlo estimation and gradient ascent-like updates, inference leverages efficient vectorized operations, automatic differentiation, just-in-time compilation, and hardware acceleration, fully implemented with [JAX](https://github.com/google/jax). 

## Quick start

The following code snippet holds an easy example that demonstrates how to use the `dibs` package.

```python
import jax
import jax.random as random
from dibs.inference import JointDiBS
from dibs.eval.target import make_nonlinear_gaussian_model
key = random.PRNGKey(0)

# simulate some data
key, subk = random.split(key)
data, model = make_nonlinear_gaussian_model(key=subk, n_vars=20, graph_prior_str="sf")

# sample 20 DAG and parameter particles from the joint posterior
dibs = JointDiBS(x=data.x, inference_model=model)
key, subk = random.split(key)
gs, thetas = dibs.sample(key=subk, n_particles=20, steps=1000)
```

## Examples
For a working example of the above, inference of the joint posterior over Gaussian Bayes net graphs and parameters, we recommend opening our example notebook in Google colab, which runs **directly from your browser**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/larslorch/dibs/blob/master/examples/dibs_joint_colab.ipynb)

Analogous notebooks can be found inside the `examples/` folder.
Executing the code as is will generate samples from the joint posterior over DAG and parameters using DiBS with SVGD and simultaneously visualize the matrices of edge probabilities modeled by the individual particles transported during inference.

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
