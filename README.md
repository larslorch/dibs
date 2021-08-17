# DiBS: Differentiable Bayesian Structure Learning

## Installation

The provided code is run with `python` on your local machine. We recommend using `anaconda`. 

The dependency requirements for reproducing the full results of the paper are more involved as the baselines of the DAG bootstrap rely on the GES and PC algorithms, for which we use the original `R` implementations via the [CausalDiscoveryToolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox). 

To install Miniconda, run

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then execute the following steps

1. After cloning the repository, cd into the root directory of the repository and run the following to create a conda environment

   ```bash
   conda create --name pardag --file requirements_conda.txt
   conda activate pardag
   ```

   

2. (For Ubuntu 18, we first need to do 

   ```bash
   sudo apt-get install gcc python3-dev
   pip3 install --user psutil --no-binary :all:
   ```

   to be able to unstall `psutils`)

   Then, run

   ```bash
   pip install -r requirements_pip.txt
   ```

   

3. Next, we need to install this code as a package as well as the `CausalDiscoveryToolbox`, which contains code for the PC and GES baselines as well as the dataset by Sachs. et al. 

   ```bash
   pip install -e .
   pip install ./CausalDiscoveryToolbox/
   ```
   
   (note the "." in the first call). That way, all scripts in (subdirectories of) the root folder will be able to access the contents properly. 
   
   The package `CausalDiscoveryToolbox` (`cdt 0.5.22`) is installed manually this way to allow fixing a bug inside their package, which caused a crash when using `ray.parallel`. 
   
   The *only* changes made to `cdt 0.5.22` are in: `cdt/utils/Settings.py:161(autoset_settings)` 

4. Lastly, if we want to run the PC or GES algorithms as baselines, we need to install all packages required for the `R` scripts underlying the `CausalDiscoveryToolbox`. For this, run

   ```bash
   sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
   sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/'
   sudo apt update
   sudo apt install r-base
   sudo apt-get install libcurl4-openssl-dev libxml2 libxml2-dev
   sudo apt install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev libv8-dev
   sudo Rscript setup.R
   ```

   where the first 3 lines allow for the installation of R version 4.0. (rather than 3.x).

   Note, to remove the `R` installation, you can run

   ```bash
   sudo apt-get purge r-base
   sudo apt-get purge r-base-core
   sudo apt autoremove
   ```

   

## Experiments

The following sections describe how to reproduce the experiments from the paper.

For each execution of `python eval.py` , the experimental results are saved inside the `results/` directory. Each method, random restart, and graph setting makes up a row inside the `.csv` file containing the obtained metrics as well as its hyperparameters.  To check that no error occurs, you can add the flag `--smoke_test` to any execution, which forces the experiment settings (iterations, random restarts, etc.) to minimal values for quick termination. When testing the code on a local machine, we recommend adding the flag `--resource_aware false`.  

### Section 6.1: Linear Gaussian BNs: Marginal Posterior Inference (Figure 2; Appendix: Figure 5)

To perform the experiment, we run the following two commands inside the directory `eval/` :

- ```python eval.py --descr "bge20" --n_vars 20 --n_variants 30 ```

- ```python eval.py --descr "bge50" --n_vars 50 --n_variants 30 ```



### Section 6.1: Linear Gaussian BNs: Joint Posterior Inference  (Figure 3; Appendix: Figure 6)

To perform the experiment, we run the following two commands inside the directory `eval/` : 

- ```python eval.py --joint --descr "joint-lingauss20" --n_vars 20 --n_variants 30 ```

- ```python eval.py --joint --descr "joint-lingauss50" --n_vars 50 --n_variants 30 ```



### Section 6.2: Nonlinear Gaussian BNs: Joint Posterior Inference  (Figure 4; Appendix: Figure 7)

To perform the experiment, we run the following commands inside the directory ` eval/` :

- ```python eval.py --joint --descr "joint-fcgauss20" --joint_inference_model fcgauss --n_vars 20 --n_variants 30 ```

- ```python eval.py --joint --descr "joint-fcgauss50-baselines" --joint_inference_model fcgauss --n_vars 50 --n_variants 30 --skip_joint_dibs ```

- ```python eval.py --joint --descr "joint-fcgauss50-dibs-n=10" --joint_inference_model fcgauss --n_vars 50 --n_variants 30 --skip_baselines --n_particles_loop 10  ```

The experiment for 50-node nonlinear Gaussian BNs is split into two runs: one for the baselines, and one for DiBS. As indicated in the caption of Figure 7, DiBS only uses 10 particles here, to make the runtime comparable.

### Section 7: Inferring Protein Signaling Networks From Single-Cell Data (Table 1; Appendix Tables 5 and 6)
To perform the experiment, we run the following commands inside the directory `eval/` :

- ```python eval.py --descr "sachs-bge" --real_data sachs --n_vars 11 --dibs_latent_dim 11 --graph_prior er --n_rollouts 30```

- ```python eval.py --descr "sachs-lingauss" --real_data sachs --n_vars 11 --joint --joint_inference_model lingauss --lingauss_obs_noise 1.0 --joint_dibs_latent_dim 11 --joint_dibs_n_grad_batch_size 100 --graph_prior er --n_rollouts 30```

- ```python eval.py --descr "sachs-fcgauss" --real_data sachs --n_vars 11 --joint --joint_inference_model fcgauss --fcgauss_obs_noise 1.0  --joint_dibs_latent_dim 11 --joint_dibs_n_grad_batch_size 100 --graph_prior er --n_rollouts 30```

To run the additional experiments that are stated in the appendix, run the above commands with the optional flags  `--bge_alpha_mu 10`  and  `--lingauss_obs_noise 0.01` (or `1.0`) and `--fcgauss_obs_noise 0.01` (or `1.0`), respectively.

### Appendix D: Additional Analyses and Ablation Studies  (Appendix Figures 8,9, and 10; Appendix Table 2)
To perform the experiments shown in the Figures, we run the following three commands inside the directory  eval/ :

- ```python eval.py --joint --descr "joint-lingauss20-er-ablation=embed" --n_vars 20 --n_variants 30  --graph_prior er --grid_joint_dibs_graph_embedding_representation --skip_baselines```

- ```python eval.py --joint --descr "joint-lingauss20-er-ablation=dim" --n_vars 20 --n_variants 30 --graph_prior er --grid_joint_dibs_latent_dim --skip_baselines```

- ```python eval.py --joint --descr "joint-lingauss20-er-ablation=steps" --n_vars 20 --n_variants 30 --graph_prior er --grid_joint_dibs_steps --skip_baselines```

To perform the experiment shown in Table 2, we run the following command inside the directory  eval/ :

- ```python marginal_edge_dist.py```

The results are printed to the command line.



### Plotting 

To generate the figures and tables  shown in the paper, we execute the following commands from inside the directory `eval/`. Each call of `plot_eval.py` returns a figure and summary table of  the corresponding experiment `.csv` data.

##### Main (Figures 2-4; Table 1):

- ```python plot_eval.py --id bge20 --path <path-to-the-results-csv> ```

- ```python plot_eval.py --id lingauss20 --path <path-to-the-results-csv> ```

- ```python plot_eval.py --id fcgauss20 --path <path-to-the-results-csv> ```

- ```python plot_eval.py --threshold_metrics --path <path-to-the-sachs-results>```

##### Appendix (Figures 5-10):

- ```python plot_eval.py --id bge50 --path <path-to-the-results-csv> ```

- ```python plot_eval.py --id lingauss50 --path <path-to-the-results-csv> ```

- ```python plot_eval.py --id fcgauss50 --path <path-to-the-results-csv> ```

- ```python plot_eval.py --ablation_embed --id lingauss20-ablation-embed --path <path-to-the-sachs-results> ```

- ```python plot_eval.py --ablation_dim --id lingauss20-ablation-dim --path <path-to-the-sachs-results> ```

- ```python plot_eval.py --ablation_steps --id lingauss20-ablation-steps --path <path-to-the-sachs-results> ```

