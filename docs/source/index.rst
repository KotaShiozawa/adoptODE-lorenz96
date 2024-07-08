.. adoptODE documentation master file, created by
   sphinx-quickstart on Wed Jun 19 13:27:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to adoptODE's documentation!
====================================

AdoptODE, introduced in [1]_, uses the library :code:`jax` [2]_ to provide easy access to the adjoint method, a powerfull tool for parameter estimation in large dynamical systems. Utilizing the auto-differentiation capabilities of :code:`jax`, most technical details necessary for the adjoint method are handled automatically, requireing the user only to input the equation of motion of the system in question. To solve the forward and backward time evolution, as default, it uses the JAX integrated odeint routine, which uses a mixed 4th/5th order Runge–Kutta scheme with Dormand–Prince adaptable
step sizing [3]_, but other JAX compatible solvers (as, for example, provided by diffrax [4]_) can be passed as option. An additional capability of adoptODE concerning the uncertainty in estimated parameters is to easily simulate data for a given system, which can be used to gauge the reliability of the recovery.

References
----------
.. [1] Lettermann, Leon, et al. Tutorial: a beginner’s guide to building a representative model of dynamical systems using the adjoint method. *Commun. phys.* **7.1** (2024): 128
.. [2] Bradbury, J. et al. JAX: Composable transformations of Python+NumPy programs (2018)
.. [3] Shampine, L. F. Some practical runge-kutta formulas. *Math. Comput.* **46**, 135–150 (1986)
.. [4] Kidger, P. On neural differential equations. Ph.D. thesis, University of Oxford (2021).