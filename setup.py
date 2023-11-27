from setuptools import setup, find_packages
setup(
    name='dibs-lib',
    version='1.3.3',
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
        'jax>=0.3.17',
        'jaxlib>=0.3.14',
        'numpy',
        'igraph',
        'imageio',
        'jupyter',
        'tqdm',
        'matplotlib',
        'scikit-learn',
    ]
)