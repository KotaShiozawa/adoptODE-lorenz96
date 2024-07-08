from setuptools import find_packages, setup

setup(
    name='adoptODE',
    version="1.0.0",
    description='Repository for "adoptODE: Fusion of data and expert knowledge for modeling dynamical systems"',
    url='https://gitlab.gwdg.de/sherzog3/adoptode.git',
    author='Leon Lettermann & Sebastian Herzog',
    author_email='sherzog3@gwdg.de',
    license='MIT',
    packages=['adoptODE'],
    install_requires=['numpy', 'jax']
)
