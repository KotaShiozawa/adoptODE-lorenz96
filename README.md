# AdoptODE
Repository for the AdoptODE Package introduced in "Tutorial: a beginner’s guide to building a representative model of dynamical systems using the adjoint method" (https://doi.org/10.1038/s42005-024-01606-9).

This repository contains the adoptODE code as well as example notebooks for setting up systems similar to those in the publication.

Documentation is also provided here: https://sherzog3.pages.gwdg.de/adoptODE/

## AdoptODE install 
------------
AdoptODE relies on [JAX](https://github.com/google/jax), for an installation of adoptODE a functioning JAX installation needs to be ensured. To use adoptODE and JAX with GPU support, which we recommend at this point, you must first install [CUDA](https://developer.nvidia.com/cuda-zone) and [CuDNN](https://developer.nvidia.com/cudnn). If you already have a working JAX installation, you can add adoptODE to your active environment by running
```bash
user@workstation:~/adoptode pip install .
```
in the repository directory.

### Conda install
------------
If this is ensured, adoptODE including JAX can be installed comparatively easily in a [conda](https://www.anaconda.com/) environment, e.g: 

#### AdoptODE with Nvidia GPU support

```bash
user@workstation:~/adoptode conda create --name adoptODE python=3.12
user@workstation:~/adoptode conda activate adoptODE
user@workstation:~/adoptode pip install --upgrade pip
user@workstation:~/adoptode pip install -r requirements_gpu.txt
```


#### AdoptODE without Nvidia GPU support

```bash
user@workstation:~/adoptode conda create --name adoptODE python=3.12
user@workstation:~/adoptode conda activate adoptODE
user@workstation:~/adoptode pip install --upgrade pip
user@workstation:~/adoptode pip install -r requirements_cpu.txt
```

### Using Apptainer/Singularity
------------
As an alternative to local installations or conda environments, you can also use the container file for [apptainer](https://apptainer.org/) or [singularity](https://sylabs.io/singularity/) , providing a minimal environment for adoptODE.

#### AdoptODE with Nvidia GPU support
For a container with GPU support run:
```bash
user@workstation:~/adoptode/container apptainer build adoptODE_GPU.sif adoptODE_GPU.def
```
After that, you can do a shell run in the container:
```bash
user@workstation:~/adoptode/container apptainer shell --nv adoptODE_GPU.sif
```
#### AdoptODE without Nvidia GPU support
To set up the container file run:
```bash
user@workstation:~/adoptode/container apptainer build adoptODE_CPU.sif adoptODE_CPU.def
```

After that, you can do a shell run in the container:
```bash
user@workstation:~/adoptode/container apptainer shell adoptODE_CPU.sif
```

## Project organization
------------

    ├── LICENSE
    ├── README.md               <- Git markdown file
    ├── data
    │   ├── SolarSystem         <- Data to reproduce the Gravitational N-body systems (Solar System) system case from the paper.
    │   └── Zebrafish           <- Subset of the zebrafish data.
    │
    ├── container               <- Folder with container defintions for apptainer/singularity
    │   ├── adoptODE_GPU.def
    │   └── adoptODE_CPU.def
    │
    ├── notebooks               <- Jupyter notebooks with exemplary implementations as they appear in the paper. Note: In some cases, the dimensions of the systems are smaller than the published values.
    │   ├── BOCF.ipynb
    │   ├── LotkaVolterra.ipynb
    │   ├── RayleighBenard.ipynb
    │   ├── RepulsiveSpheres.ipynb
    │   ├── SolarSystem.ipynb
    │   └── Zebrafish.ipynb     
    │
    ├── adoptODE                <- Sorce code folder
    │   ├── Framework.py        <- Framework defintion of adoptODE
    │   ├── ODE_Exp_dt.py       <- Adaption of the JAX included ODE solver
    │   ├── ODE_Fix_dt.py       <- Adaption of the JAX included ODE solver
    │   └── OptBounded.py       <- Customisation of some of the included optimisers in JAX
    │
    ├── requirements_gpu.txt    <- The requirements file to install adoptODE with GPU support
    ├── requirements_cpu.txt    <- The requirements file install adoptODE without GPU support
    └── setup.py                <- makes project pip installable (pip install -e .) so adoptODE can be imported

--------
