from setuptools import setup, find_packages
setup(
    name='dibs',
    version='1.0',
    description='DiBS: Differentiable Bayesian Structure Learning',
    author='Lars Lorch',
    author_email='lars.lorch@inf.ethz.ch',
    packages=find_packages(),
    install_requires=[
        'jax>=0.2.8',
        'jaxlib>=0.1.59',
        'numpy',
        'pandas',
        'python-igraph',
        'imageio',
        'jupyter',
        'tqdm',
        'matplotlib',
    ]
)
