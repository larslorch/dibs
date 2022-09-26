from setuptools import setup, find_packages
setup(
    name='dibs-lib',
    version='1.2.0',
    description='DiBS: Differentiable Bayesian Structure Learning',
    author='Lars Lorch',
    author_email='lars.lorch@inf.ethz.ch',
    url="https://github.com/larslorch/dibs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'jax>=0.2.25',
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
