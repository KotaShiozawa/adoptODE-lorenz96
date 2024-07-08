Installation
============

AdoptODE relies on `JAX <https://github.com/google/jax>`_, for an installation of adoptODE a functioning JAX installation needs to be ensured. To use adoptODE and JAX with GPU support, which we recommend at this point, you must first install `CUDA <https://developer.nvidia.com/cuda-zone>`_ and `CuDNN <https://developer.nvidia.com/cudnn>`_. If you already have a working JAX installation, you can add adoptODE to your active environment by running:

.. code-block:: shell

    user@workstation:~/adoptode$ pip install .

in the repository directory.

Conda install
-------------

If this is ensured, adoptODE including JAX can be installed comparatively easily in a `conda <https://www.anaconda.com/>`_ environment, e.g:

AdoptODE with Nvidia GPU support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    user@workstation:~/adoptode$ conda create --name adoptODE python=3.12
    user@workstation:~/adoptode$ conda activate adoptODE
    user@workstation:~/adoptode$ pip install --upgrade pip
    user@workstation:~/adoptode$ pip install -r requirements_gpu.txt

AdoptODE without Nvidia GPU support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    user@workstation:~/adoptode$ conda create --name adoptODE python=3.12
    user@workstation:~/adoptode$ conda activate adoptODE
    user@workstation:~/adoptode$ pip install --upgrade pip
    user@workstation:~/adoptode$ pip install -r requirements_cpu.txt

Using Apptainer/Singularity
---------------------------

As an alternative to local installations or conda environments, you can also use the container file for `apptainer <https://apptainer.org/>`_ or `singularity <https://sylabs.io/singularity/>`_, providing a minimal environment for adoptODE.

AdoptODE with Nvidia GPU support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a container with GPU support run:

.. code-block:: shell

    user@workstation:~/adoptode/container$ apptainer build adoptODE_GPU.sif adoptODE_GPU.def

After that, you can do a shell run in the container:

.. code-block:: shell

    user@workstation:~/adoptode/container$ apptainer shell --nv adoptODE_GPU.sif

AdoptODE without Nvidia GPU support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To set up the container file run:

.. code-block:: shell

    user@workstation:~/adoptode/container$ apptainer build adoptODE_CPU.sif adoptODE_CPU.def

After that, you can do a shell run in the container::

    user@workstation:~/adoptode/container$ apptainer shell adoptODE_CPU.sif
