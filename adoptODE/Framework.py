"""adoptODE is a universal framework to fit physical systems.

The main function of this module is generate_n_ode. Based on an equation
of mution (EOM) func and a loss function the main functions are build, 
particularly forward and gradient functions.
A second part is the class dataset_adoptODE, in which the data to be
described as well as the equations from generate_n_ode are combined. It
takes the system description as a define_system function, which returns
the EOM, a loss function and further functions to generate (at least the
shapes) of parameters and initial conditions.
If no external data should be used, the function simple_simulation generates
some data based on the EOM and parameters and initial conditions generated
by define_function. Hence, for this case the parameters and initial
conditions should be meaninful.
Parameters of the EOM are distinguished between three categories:

1. params: The normal general parameters of the system which are unknown.

2. iparams: Also unknown parameters, which might depend on the
    realization, e.g. if for example multiple realizations are
    considered simultaneously, params are one set of parameters
    to fit all of them, while iparams, short for individual
    parameters, are seperatly fitted for each realization.

3. exparams: Parameters which are known, but cannot be build into the
    systems definition because they might be different between
    different observations.

All types of parameters as well as the systems state can be structured using
JAX PyTrees, allowing for arbitrary nested combinations of Dictionaries,
Lists, Tuples and Arrays.

A dummy define_system function is provided below. Then the workflow is the
following:

.. code-block:: python

    kwargs_sys = {} # Dictionary containing system parameters
    kwargs_adoptODE = {'lr':0.01, 'epochs':100} # Necessary hyperparameters for adoptODE
    t_evals = array # times at which you observed your system

        
For fitting Data
----------------

(for details see :class:`dataset_adoptODE`)

.. code-block:: python

    ys = Load data you want to fit
    my_dataset = dataset_adoptODE(define_system,
                                ys,
                                t_evals,
                                kwargs_sys,
                                kwargs_adoptODE)
    train_adoptODE(my_dataset)

For simulating Data
-------------------

(for details see :func:`simple_simulation`)

.. code-block:: python

    my_dataset = simple_simulation(define_system,
                                t_evals,
                                kwargs_sys,
                                kwargs_adoptODE)
    train_adoptODE(my_dataset)
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax.experimental import ode
from jax.example_libraries import optimizers
from jax import lax
from jax import tree_util
from jax import flatten_util
from functools import partial
from typing import Any, Tuple
from . import OptBounded

PyTree = Any


def define_system_dummy(**kwargs_sys):
    """Example of a very simple system following the equation
    :math:`\mathrm{d}y/\mathrm{d}t = a\cdot y + b`

    An example of a simple linear system with only to parameters a and b, of which
    a ist treated as param and b as iparam. The advantaged of this system is that
    it can be solved analytically (even the adoptODE internal augmented ODE system),
    which makes it useful for testing. For implementin another system you can
    copy this function and modify it to represent your system.

    Args:
        kwargs_sys: Dictionary of parameters defining the system, in this case only
            the number of realizations, that is parallel systems with different b but
            same a, :code:`N_sys`.

    Returns:
        [Callable, ..., dict]: A number of functions specific to the system in question:
            - :code:`eom`: The system's EOM
            - :code:`loss`: The loss function
            - :code:`gen_params`: The three different types of parameters
            - :code:`gen_y0`: The initial conditions (For a single realization!)
            - :code:`{}`: A dictionary of internal functions, for example building up a
                more complicated EOM, which should be accessible later for testing.
    """

    def gen_params() -> Tuple[PyTree, PyTree, PyTree]:
        """Defines the system's parameters.

        In case only Data should be fitted, only the shapes of the returned
        PyTrees are relevant, and the returned values might only be used as
        initial guess for the parameters. In case of simulations the values
        should be meaningful.

        Returns:
            [PyTree, ...]: Three PyTrees, the first beeing the params, the second iparams
                and the third exparams. Only here for iparams and exparams the
                leading axis over different realizations have to be manually
                included.
        """
        return {"a": np.random.rand()}, {"b": np.random.rand(kwargs_sys["N_sys"])}, {}

    def gen_y0():
        """Defines the system's initial conditions.

        Returns:
            PyTree: saving the state of a single system at a single time.
        """
        return {"State": np.random.rand()}

    @jax.jit
    def eom(
        y: PyTree, _t: float, params: PyTree, iparams: PyTree, _exparams: PyTree
    ) -> PyTree:
        """The Equation of Motion (EOM).

        The EOM is always defined for a single realization, meaning y, iparams
        and exparams do not have a leading axis for different system realizations
        here.

        Args:
            y: The system state at current time as PyTree.
            _t: the current time, unused here this EOM is not explicitly
                time dependend.
            params: The normal general parameters of the system which are
                unknown.
            iparams: Also unknown parameters, which might depend on the
                realization, e.g. if for example multiple realizations are
                considered simultaneously, params are one set of parameters
                to fit all of them, while iparams, short for individual
                parameters, are seperatly fitted for each realization.
            exparams: Parameters which are known, but cannot be build into the
                systems definition because they might be different between
                different observations. Unused in this example.

        Returns:
          PyTree: of same shape as y, with every element giving the time derivative
          of this precise element in y.
        """
        return {"State": params["a"] * y["State"] + iparams["b"]}

    @jax.jit
    def loss(
        ys: PyTree,
        _params: PyTree,
        _iparams: PyTree,
        _exparams: PyTree,
        ys_targets: PyTree,
    ) -> float:
        """Defining the loss to be minimized.

        Args:
            ys: The systems state at all times specified in t_evals.
            params, iparams, exparams: as above in eom()
            ys_targets: Expected results for this realization, PyTree
                        of the same shape as ys.

        Returns:
            The loss function value as float.
        """
        return jnp.sum(
            (
                flatten_util.ravel_pytree(ys)[0]
                - flatten_util.ravel_pytree(ys_targets)[0]
            )
            ** 2
        )

    return eom, loss, gen_params, gen_y0, {}


def _select_ode_solver(**kwargs):
    """Selects the ODE solver

    Args:
        **kwargs: May take the full kwarg list, relevant are
            only :code:`dt` and :code:`ODE_solver`.

    Returns:
        ODE solver functions for forward and backward pass.
    """
    if "dt" in kwargs:
        if "ODE_solver" in kwargs:

            def odeint(f, y, ts, *args, rtol=1e-5, atol=1e-5):
                return kwargs["ODE_solver"](
                    f, y, ts, kwargs["dt"], *args, rtol=rtol, atol=atol
                )

            def odeint_back(f, y, ts, *args, rtol=1e-5, atol=1e-5):
                return kwargs["ODE_solver"](
                    f, y, ts, -kwargs["dt"], *args, rtol=rtol, atol=atol
                )

        else:
            raise Exception(
                "If timestep dt is specified, a fixed timestep ODE solver "
                "has to be passed via kwargs_adoptODE's 'ODE_solver' keyword."
                "(Sover gets negative dt to determin its backward pass, but"
                "should use as if positive!)"
            )
    else:
        if "ODE_solver" in kwargs:

            def odeint(f, y, ts, *args, rtol=1e-5, atol=1e-5):
                return kwargs["ODE_solver"](f, y, ts, *args, rtol=rtol, atol=atol)

            odeint_back = odeint
        else:

            def odeint(f, y, ts, *args, rtol=1e-5, atol=1e-5):
                return ode.odeint(f, y, ts, *args, rtol=rtol, atol=atol)

            odeint_back = odeint
    return odeint, odeint_back


class EquationsNODE:
    """Generates gradient function for EOM and loss function via adoptODE.

    This is the core routine of the module. Given an EOM and a loss function,
    the adoptODE method is used to build a function computing the gradients of the
    given loss function with respect to the parameters of the EOM. The theory of
    adoptODEs is given in [1]_
    The power of this routine is to derive the full necessary framework,
    including jacobians of the EOM and the loss function without further ado by
    the user.
    If not otherwise stated, the arguments in the subroutines always have the
    following meaning:

    - :code:`y0`: Initial state of a single system.
    - :code:`y`: current state of a single system.
    - :code:`params`: Set of parameters of a single system. (See :func:`define_system_dummy`)
    - :code:`iparams`: Set of iparams for a single system. (See :func:`define_system_dummy`)
    - :code:`exparams`: Set of exparams for a single system. (See :func:`define_system_dummy`)
    - :code:`t_evals`: The time this system is evaluated.
    - :code:`ys`: The state of a system at all times in t_eval, the timepoints being the leading axis.
    - :code:`ys_targets`: Same as ys, but not results of the current model, but the provided data to be fitted.

    Args:
        eom: EOM as given in :func:`define_system_dummy` above.
        loss: loss function as given in :func:`define_system_dummy` above
        kwargs_adoptODE: Hyperparameters for adoptODE, for details see :class:`dataset_adoptODE`

    Returns: v_forward, gradient_adjoint_batch, back_check, grad_check, v_loss
        - :code:`v_forward`: Vectorized forward function for a batch of systems
        - :code:`gradient_adjoint_batch`: The main routine, yielding the gradients of a batch of systems with given target values.
        - :code:`backwards_check`: A test function to check if the backwards pass is stable in terms of the occuring ODE solutions of the system.
        - :code:`gradient_check`: A test function to check if the backwards pass is stable in terms of the occuring ODE solutions to the gradients and additional dimensions.
        - :code:`v_loss`: A vectorized version of loss, computing the loss for a batch of systems.

    References:
    -----------
    .. [1] Chen, Ricky TQ, et al. "Neural ordinary differential equations." arXiv preprint arXiv:1806.07366 (2018).
    """

    def __init__(self, eom, loss, **kwargs_adoptODE):
        odeint, odeint_back = _select_ode_solver(**kwargs_adoptODE)
        self._odeint, self._odeint_back = odeint, odeint_back

        # Selecting wheter to vecotrize t_evals over 0 axis or not at all.
        t_evals_vec_axis = 0 if kwargs_adoptODE["vectorize_t_evals"] is True else None

        @jax.jit
        def forward(y0, t_evals, params, iparams, exparams):
            """Defines the standard forward pass for a single system, returning ys."""
            ys = odeint(
                eom,
                y0,
                t_evals,
                params,
                iparams,
                exparams,
                rtol=kwargs_adoptODE["rtol"],
                atol=kwargs_adoptODE["atol"],
            )
            return ys

        self._forward = forward

        @partial(jax.jit, static_argnums=(6, 7))
        def r_forward(
            y0, t_evals, params, iparams, exparams, ys_target, t_stop_idx, t_reset_idcs
        ):
            """A modified forward function if t_reset_idxs mechanics is active.

            This foward function is propagating forward, but at every time in t_evals
            indicated by its index in t_reset_idxs_ext, the system state is reset to
            the state indicated in the provided data, ys_target, for that time.

            """
            ys = tree_util.tree_map(lambda x: jnp.nan * jnp.zeros_like(x), ys_target)
            first_idx = t_reset_idcs[0]
            ys = tree_util.tree_map(
                lambda x, y: x.at[first_idx].set(y[first_idx]), ys, y0
            )
            state = (ys, params, iparams, exparams, y0)
            t_start_idcs = t_reset_idcs
            t_end_idcs = t_reset_idcs[1:] + (t_stop_idx,)
            max_slice_size = max([b - a for a, b in zip(t_start_idcs, t_end_idcs)])

            def r_forward_body(i, state):
                ys, params, iparams, exparams, y0 = state

                t_start, t_end = jnp.array(t_start_idcs)[i], jnp.array(t_end_idcs)[i]
                y0_i = tree_util.tree_map(lambda x: x[i], y0)
                t_evals_ext = jnp.concatenate(
                    (t_evals, jnp.ones(max_slice_size + 1, int) * t_evals[-1])
                )
                t_evals_i = jnp.where(
                    np.arange(max_slice_size + 1) <= (t_end - t_start),
                    jax.lax.dynamic_slice(t_evals_ext, [t_start], [max_slice_size + 1]),
                    t_evals[t_end],
                )
                ys_new = odeint(
                    eom,
                    y0_i,
                    t_evals_i,
                    params,
                    iparams,
                    exparams,
                    rtol=kwargs_adoptODE["rtol"],
                    atol=kwargs_adoptODE["atol"],
                )

                update_a_time_state = (ys, ys_new, t_start)

                def update_a_time(idx, update_state):
                    ys, ys_new, t_start = update_state
                    ys = tree_util.tree_map(
                        lambda x, y: x.at[t_start + idx].set(y[idx]), ys, ys_new
                    )
                    return (ys, ys_new, t_start)

                ys = jax.lax.fori_loop(
                    1, t_end - t_start + 1, update_a_time, update_a_time_state
                )[0]

                return (ys, params, iparams, exparams, y0)

            ys = lax.fori_loop(0, len(t_reset_idcs), r_forward_body, state)[0]
            return ys

        self._r_forward = r_forward

        @jax.jit
        def v_forward(y0_batch, t_evals, params, iparams_batch, exparams_batch):
            """Vectorized forward function, dealing with a batch of systems."""
            return jax.vmap(forward, in_axes=(0, t_evals_vec_axis, None, 0, 0))(
                y0_batch, t_evals, params, iparams_batch, exparams_batch
            )

        self._v_forward = v_forward

        @partial(jax.jit, static_argnums=(6, 7))
        def v_r_forward(
            y0_batch,
            t_evals,
            params,
            iparams_batch,
            exparams_batch,
            ys_target_batch,
            t_stop_idx,
            t_reset_idcs,
        ):
            """Vectorized r_forward function, dealing with a batch of systems."""
            return jax.vmap(r_forward, in_axes=(0, None, None, 0, 0, 0, None, None))(
                y0_batch,
                t_evals,
                params,
                iparams_batch,
                exparams_batch,
                ys_target_batch,
                t_stop_idx,
                t_reset_idcs,
            )

        self._v_r_forward = v_r_forward

        @jax.jit
        def dloss_dy(ys, index, params, iparams, exparams, ys_target):
            """Jacobian of loss w.r.t. y at time t_evals[index]."""
            return tree_util.tree_map(
                lambda x: jnp.asarray(x)[index],
                jax.grad(lambda ys_l: loss(ys_l, params, iparams, exparams, ys_target))(
                    ys
                ),
            )

        self._dloss_dy = dloss_dy

        @jax.jit
        def dloss_dp(ys, params, iparams, exparams, ys_target):
            """Jacobian of loss w.r.t. params."""
            return jax.grad(
                lambda params: loss(ys, params, iparams, exparams, ys_target)
            )(params)

        self._dloss_dp = dloss_dp

        @jax.jit
        def dloss_dip(ys, params, iparams, exparams, ys_target):
            """Jacobian of loss w.r.t. iparams."""
            return jax.grad(
                lambda iparams: loss(ys, params, iparams, exparams, ys_target)
            )(iparams)

        self._dloss_dip = dloss_dip

        if kwargs_adoptODE["multi_measurement_constraint"] is None:

            @jax.jit
            def multi_meas_grad(ys, params, iparams, exparams, ys_target):
                return tree_util.tree_map(lambda x: 0, iparams)

        else:

            @jax.jit
            def multi_meas_grad(ys, params, iparams, exparams, ys_target):
                return jax.grad(
                    lambda iparams: kwargs_adoptODE["multi_measurement_constraint"](
                        ys, params, iparams, exparams, ys_target
                    )
                )(iparams)

        self.multi_meas_grad = multi_meas_grad

        @jax.jit
        def v_loss(ys_batch, params, iparams_batch, exparams_batch, ys_target_batch):
            """Vectorized loss function handling a batch of systems."""
            return jax.vmap(loss, in_axes=(0, None, 0, 0, 0))(
                ys_batch, params, iparams_batch, exparams_batch, ys_target_batch
            )

        self._v_loss = v_loss
        # Augmented dynamic of adoptODE is defined, either with the solver timestep as
        # additional argument or without, the latter is the standard case. In both
        # cases the dynamic is reversed in defining the expression, such that the
        # ODE solver is not required to run backwards.
        if kwargs_adoptODE["EOM_dt_dependent"] is True:

            @jax.jit
            def aug_dyn(aug_dyn_state, t_fake, dt, params, iparams, t2, exparams):
                """The eom of the augmented dynamic of the eom.

                This augmented system has to be solved backwards. As not all ODE solvers
                allow for negative time propagation, this function is build such that
                solving it forward corresponst do solving the true augmented system back-
                wards. The true time of the system is given by the end of the interval in
                question, t2, minus the fake forward propagating time t_fake.

                Args:
                    aug_dyn_state: State of the augmented system. This is a tuple:
                        (system state, adjoinet state, gradients params, gradients iparams)
                    t_fake: Forward counting time from zero, mapped to run backwards.
                    dt: If 'EOM_dt_dependent' is set, the timestep of the solver
                    t2: End of the interval solved for, real time is t2-t_fake

                Returns:
                    The negative time derivative of the augmented state, having the same
                        shape as aug_dyn_state.
                """
                t = t2 - t_fake
                y = aug_dyn_state[0]
                a = aug_dyn_state[1]
                Jy_vjp = jax.vjp(lambda y: eom(y, t, dt, params, iparams, exparams), y)[
                    1
                ]
                Jp_vjp = jax.vjp(
                    lambda params: eom(y, t, dt, params, iparams, exparams), params
                )[1]
                Jip_vjp = jax.vjp(
                    lambda iparams: eom(y, t, dt, params, iparams, exparams), iparams
                )[1]
                Jy = Jy_vjp(a)[0]
                Jp = Jp_vjp(a)[0]
                Jip = Jip_vjp(a)[0]
                return (
                    tree_util.tree_map(
                        lambda x: -x, eom(y, t, dt, params, iparams, exparams)
                    ),
                    Jy,
                    Jp,
                    Jip,
                )

        else:

            @jax.jit
            def aug_dyn(aug_dyn_state, t_fake, params, iparams, t2, exparams):
                """Identical to function above, but without dependence on stepsize dt."""
                t = t2 - t_fake
                y = aug_dyn_state[0]
                a = aug_dyn_state[1]
                Jy_vjp = jax.vjp(lambda y: eom(y, t, params, iparams, exparams), y)[1]
                Jp_vjp = jax.vjp(
                    lambda params: eom(y, t, params, iparams, exparams), params
                )[1]
                Jip_vjp = jax.vjp(
                    lambda iparams: eom(y, t, params, iparams, exparams), iparams
                )[1]
                Jy = Jy_vjp(a)[0]
                Jp = Jp_vjp(a)[0]
                Jip = Jip_vjp(a)[0]
                return (
                    tree_util.tree_map(
                        lambda x: -x, eom(y, t, params, iparams, exparams)
                    ),
                    Jy,
                    Jp,
                    Jip,
                )

        self._aug_dyn = aug_dyn

        # Following four functions define two nested loops. The outer (backward_loop
        # and backward_step) does one iteration for every interval going back from
        # one time in t_evals to the previous one.
        # The inner (backup_loop and backup_step) subdivides this interval further in
        # as many substeps as given by the 'N_backups' keyword, and at every substep
        # reloads the systems state saved during a forward pass. This is to mitigate
        # problems with stability while solving the system backwards, e.g. in
        # diffusive systems.

        @jax.jit
        def backup_body(i, state):
            """The body function of the backup loop.

            Args:
                i: The loop counter variable.
                state: The loop state, being a tuple consiting of the following:
                    aug_dyn_state: the initial state of the augmented system for this step.
                    length_backup: duration of one packup step.
                    params, iparams exparams as always.
                    t_backups: The times at which backups from forward pass should be used.
                    backups: The backups save during forward pass.
                    time_decay: Additional hyperparameter, to set of exponential decay of
                                gradients over time, controlled by the 'time_decay' keyword
                                in kwargs_adoptODE.

            Returns:
                The updated state after one backup interval was propagated through.
            """
            (
                aug_dyn_state,
                length_backup,
                params,
                iparams,
                t_backups,
                exparams,
                backups,
                time_decay,
            ) = state
            n_backups = kwargs_adoptODE["N_backups"]
            sol = odeint_back(
                aug_dyn,
                aug_dyn_state,
                jnp.array([0.0, length_backup]),
                params,
                iparams,
                t_backups[n_backups - i],
                exparams,
                rtol=kwargs_adoptODE["rtol"],
                atol=kwargs_adoptODE["atol"],
            )
            aug_dyn_state = (
                tree_util.tree_map(lambda x: x[n_backups - i - 1], backups),
                *(
                    tree_util.tree_map(
                        lambda x: x[-1] * time_decay**length_backup, sol[1:]
                    )
                ),
            )
            return (
                aug_dyn_state,
                length_backup,
                params,
                iparams,
                t_backups,
                exparams,
                backups,
                time_decay,
            )

        self._backup_body = backup_body

        @jax.jit
        def backup_loop(
            y_t2, y_t1, a_t2, params, iparams, exparams, ts, dLdp, dLdip, time_decay
        ):
            """The loop function for the backup_body function defined above.

            Solving the augmented system from the first to the second, earlier time
            provided via in ts. This is done by subdividing the interval into smaller
            ones equal in number to kwargs_adoptODE['N_backups']. First a forward pass is
            used to save the solution of the system at every backup time, afterwards
            these are used by backup_body to stabilize backward solution.

            Args:
                y_t2: System state at late time t2
                y_t1: System state at early time t1
                a_t2: adjoint state at late time t2
                ts: [t2, t1]
                dLdp: Gradient of loss w.r.t. params
                dLdip: Gradient of loss w.r.t. iparams
                time_decay: Gradient and adjoint state decay over time.

            Returns:
                The state of the augmented dynamics after backpropagation from t2 to t1.
            """
            n_backups = kwargs_adoptODE["N_backups"]
            t_backups = jnp.linspace(ts[1], ts[0], n_backups + 1)
            length_backup = t_backups[1] - t_backups[0]
            if n_backups > 1:
                backups = forward(y_t1, t_backups, params, iparams, exparams)
            else:
                backups = tree_util.tree_map(lambda x, y: jnp.stack((x, y)), y_t1, y_t2)
            aug_dyn_state = (y_t2, a_t2, dLdp, dLdip)
            state_init = (
                aug_dyn_state,
                length_backup,
                params,
                iparams,
                t_backups,
                exparams,
                backups,
                time_decay,
            )
            state_final = lax.fori_loop(0, n_backups, backup_body, state_init)
            return state_final[0]

        self._backup_loop = backup_loop

        @jax.jit
        def backwards_body(i, state):
            """The body function of the backwards loop.

            Args:
                i: The loop counter variable.
                state: The loop state, being a tuple consiting of standard arguments and
                the following:
                    dLdp: Gradient of loss w.r.t. params
                    dLdip: Gradient of loss w.r.t. iparams
                    a_old: Initial value of adjoint state
                    t2_index: Index within t_evals of the late time boundary of the
                    interval concerned.
                    time_decay: Gradient and adjoint state decay over time

            Returns: The state after passing backwards one interval between to times
                in t_evals.
            """
            (
                dLdp,
                dLdip,
                a_old,
                ys,
                params,
                iparams,
                exparams,
                t_evals,
                ys_target,
                t2_index,
                time_decay,
            ) = state
            index = t2_index - i
            y_t2 = tree_util.tree_map(lambda x: x[index], ys)
            y_t1 = tree_util.tree_map(lambda x: x[index - 1], ys)
            a_t2 = tree_util.tree_map(
                lambda x, y: jnp.nan_to_num(x) + y,
                dloss_dy(ys, index, params, iparams, exparams, ys_target),
                a_old,
            )
            sol = backup_loop(
                y_t2,
                y_t1,
                a_t2,
                params,
                iparams,
                exparams,
                t_evals[jnp.array([index, index - 1])],
                dLdp,
                dLdip,
                time_decay,
            )
            dLdp = sol[2]
            dLdip = sol[3]
            a_old = sol[1]
            return (
                dLdp,
                dLdip,
                a_old,
                ys,
                params,
                iparams,
                exparams,
                t_evals,
                ys_target,
                t2_index,
                time_decay,
            )

        self._backwards_body = backwards_body

        @jax.jit
        def backwards_loop(
            ys, t_eval, params, iparams, exparams, ys_target, time_decay, t_target_range
        ):
            """The loop function for propagating backwards.

            Args:
                standard arguments explained above, and
                t_target_range: The two indices of times in t_eval between which (in
                                total!) gradients should be computed. This my not be
                                the complete range if either 't_stop_idx' or
                                't_reset_idxs' keywords are active.

            Returns:
                dLdp: The gradients w.r.t. params
                dLdip: The gradients w.r.t. iparams
                a: The adjoined state at beginning of the concerned interval.
                    If this is at the initial time, this result is the gradients
                    with respect to initial conditions!
            """
            dLdp, dLdip = tree_util.tree_map(
                jnp.zeros_like, params
            ), tree_util.tree_map(jnp.zeros_like, iparams)
            a_old = tree_util.tree_map(lambda x: jnp.zeros(x.shape[1:]), ys)
            state_init = (
                dLdp,
                dLdip,
                a_old,
                ys,
                params,
                iparams,
                exparams,
                t_eval,
                ys_target,
                t_target_range[1],
                time_decay,
            )
            state_final = lax.fori_loop(
                0, t_target_range[1] - t_target_range[0], backwards_body, state_init
            )
            return state_final[0], state_final[1], state_final[2]

        self._backwards_loop = backwards_loop

        @partial(jax.jit, static_argnums=(6, 8))
        def gradient_adjoint(
            y0,
            t_evals,
            params,
            iparams,
            exparams,
            ys_target,
            t_stop_idx=None,
            time_decay=1.0,
            t_reset_idcs=None,
        ):
            """Computing the gradient for a single system.

            Args:
                y0: Initial state of a single system.
                t_evals: The time this system is evaluated.
                params: Set of parameters of the system. (See define_system_dummy)
                iparams: Set of iparams for a single system. (See define_system_dummy)
                exparams: Set of exparams for a single system. (See define_system_dummy)
                ys_targets: The provided data to be fitted, for a single system.
                t_stop_idx: Times past t_evals(t_stop_idx) are not contributing to grads
                time_decay: Decay of gradients with time, meaning gradients caused by
                    behaviour further from initial time have less influence if
                    this is <1.
                t_reset_idxs: Instead of propagating the system from some initial
                    condition, here at multiple times (the reset times) specified
                    by their indices in t_evals the target value at that time is
                    used as a fresh start value. Useful in chaotic dynamics to
                    prevent the system inevitably diverting from target behavior
                    for longer times.

            Returns:
                dLdp: The gradients w.r.t. params
                dLdip: The gradients w.r.t. iparams
                dLdy0: The gradients w.r.t. the initial conditions
            """
            if t_stop_idx is None and t_reset_idcs is None:
                ys = forward(y0, t_evals, params, iparams, exparams)
                t_idx_range = jnp.array([[0, t_evals.shape[-1] - 1]])
                y0_to_take = 0
            else:
                if t_stop_idx is None:
                    t_stop_idx = t_evals.shape[-1] - 1
                if t_reset_idcs is None:
                    t_reset_idcs = (0,)
                t_reset_idcs_ext = t_reset_idcs + (t_stop_idx,)
                t_idx_range = jnp.array(
                    [
                        [t_reset_idcs_ext[i], t_reset_idcs_ext[i + 1]]
                        for i in range(len(t_reset_idcs))
                    ]
                )
                ys = r_forward(
                    y0,
                    t_evals,
                    params,
                    iparams,
                    exparams,
                    ys_target,
                    t_stop_idx,
                    t_reset_idcs,
                )
                y0_to_take = jnp.arange(len(t_reset_idcs))
            # Vectorization for different reset initial times
            dLdp, dLdip, dLdy0 = jax.vmap(
                backwards_loop,
                in_axes=(None, None, None, None, None, None, None, 0),
                out_axes=(0, 0, 0),
            )(
                ys,
                t_evals,
                params,
                iparams,
                exparams,
                ys_target,
                time_decay,
                t_idx_range,
            )

            dLdp = tree_util.tree_map(
                lambda x, y: jnp.sum(x, axis=0) + jnp.nan_to_num(y),
                dLdp,
                dloss_dp(ys, params, iparams, exparams, ys_target),
            )
            dLdip = tree_util.tree_map(
                lambda x, y: jnp.sum(x, axis=0) + jnp.nan_to_num(y),
                dLdip,
                dloss_dip(ys, params, iparams, exparams, ys_target),
            )
            dLdy0 = tree_util.tree_map(
                lambda x, y: x[y0_to_take] + jnp.nan_to_num(y),
                dLdy0,
                dloss_dy(ys, y0_to_take, params, iparams, exparams, ys_target),
            )
            return dLdp, dLdip, dLdy0, ys

        self._gradient_adjoint = gradient_adjoint

        @partial(jax.jit, static_argnums=(6, 8))
        def gradient_adjoint_batch(
            y0_batch,
            t_evals,
            params,
            iparams_batch,
            exparams_batch,
            ys_target_batch,
            t_stop_idx=None,
            time_decay=1.0,
            t_reset_idcs=None,
        ):
            """Computing the gradient for a batch of systems.

            Args:
                y0: Initial state of a batch of systems.
                t_evals: The time this system is evaluated.
                params: Set of parameters of the system. (See define_system_dummy)
                iparams: Set of iparams for batch of systems. (See define_system_dummy)
                exparams: Set of exparams for batch of systems. (See define_system_dummy)
                ys_targets_batch: The provided data to be fitted, for a batch of systems.
                t_stop_idx: Times past t_evals(t_stop_idx) are not contributing to grads
                time_decay: Decay of gradients with time, meaning gradients caused by
                    behaviour further from initial time have less influence if
                    this is <1.
                t_reset_idxs: Instead of propagating the system from some initial
                    condition, here at multiple times (the reset times) specified
                    by their indices in t_evals the target value at that time is
                    used as a fresh start value. Useful in chaotic dynamics to
                    prevent the system inevitably diverting from target behavior
                    for longer times.

            Returns:
                dLdp: The gradients w.r.t. params, averaged over all system
                dLdip: The gradients w.r.t. iparams, individually for each system
                dLdy0: The gradients w.r.t. the initial conditions, individually for
                    each system.
            """
            dLdp, dLdip, dLdy0, ys_batch = jax.vmap(
                gradient_adjoint,
                in_axes=(
                    0,
                    t_evals_vec_axis,
                    None,
                    0,
                    0,
                    0,
                    t_evals_vec_axis,
                    t_evals_vec_axis,
                    None,
                ),
                out_axes=(0, 0, 0, 0),
            )(
                y0_batch,
                t_evals,
                params,
                iparams_batch,
                exparams_batch,
                ys_target_batch,
                t_stop_idx,
                time_decay,
                t_reset_idcs,
            )  # jit removed
            dLdp = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), dLdp)
            dLdip = tree_util.tree_map(
                lambda x, y: x + y,
                dLdip,
                multi_meas_grad(
                    ys_batch, params, iparams_batch, exparams_batch, ys_target_batch
                ),
            )
            return dLdp, dLdip, dLdy0, ys_batch

        self._gradient_adjoint_batch = gradient_adjoint_batch

    def forward(self, y0_batch, t_evals, params, iparams_batch, exparams_batch):
        return self._v_forward(y0_batch, t_evals, params, iparams_batch, exparams_batch)

    def resetted_forward(
        self,
        y0_batch,
        t_evals,
        params,
        iparams_batch,
        exparams_batch,
        ys_target_batch,
        t_stop_idx,
        t_reset_idcs,
    ):
        t_r_i_here = (0,) if t_reset_idcs is None else t_reset_idcs
        t_s_i_here = t_evals.shape[-1] - 1 if t_stop_idx is None else t_stop_idx
        return self._v_r_forward(
            y0_batch,
            t_evals,
            params,
            iparams_batch,
            exparams_batch,
            ys_target_batch,
            t_s_i_here,
            t_r_i_here,
        )

    def gradient(
        self,
        y0_batch,
        t_evals,
        params,
        iparams_batch,
        exparams_batch,
        ys_target_batch,
        t_stop_idx=None,
        time_decay=1.0,
        t_reset_idcs=None,
    ):
        return self._gradient_adjoint_batch(
            y0_batch,
            t_evals,
            params,
            iparams_batch,
            exparams_batch,
            ys_target_batch,
            t_stop_idx=t_stop_idx,
            time_decay=time_decay,
            t_reset_idcs=t_reset_idcs,
        )

    def loss(self, ys_batch, params, iparams_batch, exparams_batch, ys_target_batch):
        return self._v_loss(
            ys_batch, params, iparams_batch, exparams_batch, ys_target_batch
        )


def pytree_norm(a, b=None):
    """Computes the L2 norm for one or for the difference of two PyTrees."""
    if b is None:
        return np.sqrt(np.sum(flatten_util.ravel_pytree(a)[0] ** 2))
    else:
        diff = tree_util.tree_map(lambda x, y: x - y, a, b)
        return np.sqrt(sum(flatten_util.ravel_pytree(diff)[0] ** 2))


def pytree_to_device(tree):
    """Sends a pytree to a detected GPU device for computation."""
    return tree_util.tree_map(jax.device_put, tree)


def pytree_select_system(tree, system):
    """Selects one system out of the batch of systems"""
    return tree_util.tree_map(lambda x: x[system], tree)


def pytree_select_time(tree, time):
    """Selects one time for all systems."""
    return tree_util.tree_map(lambda x: x[:, time], tree)


def build_optimizer(lr, lr_decay, optimizer, opt_kwargs, custom_scheduel=None):
    """Building the adam optimizer
    Using entries from kwargs_adoptODE parameters and learning rate schedule
    are set up"""
    if custom_scheduel is None:  # ADD TO DOCU!
        if lr_decay is None:
            opt_init, opt_update, get_params = optimizer(lr, **opt_kwargs)
        else:
            opt_init, opt_update, get_params = optimizer(
                optimizers.exponential_decay(lr, 1, lr_decay), **opt_kwargs
            )
    else:
        opt_init, opt_update, get_params = optimizer(custom_scheduel, **opt_kwargs)
    return opt_init, opt_update, get_params


def to_np_arr(x):
    """Convert Arrays in PyTree back to Numpy Arrays."""
    return tree_util.tree_map(np.array, x)


def train_adoptODE(
    dataset, print_interval="Auto", save_interval=None, print_both_losses=False
):
    """Function to perform training for prepared :class:`dataset_adoptODE` object.

    No specification of training can be made while calling this function,
    everything has to be set out within the passed dataset object. The dataset
    object is also modified by this and the new trained values for parameters
    are updated.

    Args:
        dataset: The dataset object which should be trained.
        print_interval: Training status is printed every {this} epochs.
        save_interval:  Training status is saved every {this} epochs.

    Returns:
      1. The final set of parameters/initial conditions as dictionary.
      2. A list of the losses saved during training.
      3. A list of the errors of params, iparams and y0. Only meaningful if
          truth values for theses parameters have been included in the dataset
          object, as is the case when the data was generated by simulation.
      4. A history, saving the parameters at every epoch indicated by
          save_interval.
    """
    kwargs_adoptODE = dataset.kwargs_adoptODE
    epochs, t_stop_idx, time_decay, t_reset_idcs = (
        kwargs_adoptODE["epochs"],
        kwargs_adoptODE["t_stop_idx"],
        kwargs_adoptODE["time_decay"],
        kwargs_adoptODE["t_reset_idcs"],
    )

    if print_interval == "Auto":
        print_interval = np.maximum(int(epochs / 10), 1)
    if save_interval == "Auto":
        save_interval = np.maximum(int(epochs / 20), 1)
    if print_interval is None:
        print_interval = np.nan
    if save_interval is None:
        save_interval = np.nan
    t_evals = jnp.array(dataset.t_evals)
    params_true = pytree_to_device(dataset.params)
    iparams_true = pytree_to_device(dataset.iparams)
    exparams = pytree_to_device(dataset.exparams)
    y0 = pytree_to_device(dataset.y0_train)
    ys_target = pytree_to_device(dataset.ys)
    losses = []
    errors = []
    params_history = []

    grad_func = dataset.gradient
    loss_func = dataset.loss
    forward_func = dataset.forward
    resetted_forward_func = dataset.resetted_forward
    params_train = pytree_to_device(dataset.params_train)
    iparams_train = pytree_to_device(dataset.iparams_train)

    opt_init, opt_update, get_params = build_optimizer(
        kwargs_adoptODE["lr"],
        kwargs_adoptODE["lr_decay"],
        kwargs_adoptODE["optimizer"],
        kwargs_adoptODE["optimizer_kwargs"],
        kwargs_adoptODE["custom_scheduel"],
    )
    opt_state = opt_init(
        params_train,
        lower_bound=kwargs_adoptODE["lower_b"],
        upper_bound=kwargs_adoptODE["upper_b"],
    )

    opt_init_ip, opt_update_ip, get_iparams = build_optimizer(
        kwargs_adoptODE["lr_ip"],
        kwargs_adoptODE["lr_decay_ip"],
        kwargs_adoptODE["optimizer_ip"],
        kwargs_adoptODE["optimizer_ip_kwargs"],
        kwargs_adoptODE["custom_scheduel_ip"],
    )
    opt_state_ip = opt_init_ip(
        iparams_train,
        lower_bound=kwargs_adoptODE["lower_b_ip"],
        upper_bound=kwargs_adoptODE["upper_b_ip"],
    )

    opt_init_y0, opt_update_y0, get_y0 = build_optimizer(
        kwargs_adoptODE["lr_y0"],
        kwargs_adoptODE["lr_decay_y0"],
        kwargs_adoptODE["optimizer_y0"],
        kwargs_adoptODE["optimizer_y0_kwargs"],
        kwargs_adoptODE["custom_scheduel_y0"],
    )
    opt_state_y0 = opt_init_y0(
        y0,
        lower_bound=kwargs_adoptODE["lower_b_y0"],
        upper_bound=kwargs_adoptODE["upper_b_y0"],
    )

    def losses_for_print(opt_state, opt_state_ip, opt_state_y0, ys_from_grad=None):
        loss_string = ""
        if (t_reset_idcs is None and t_stop_idx is None) or print_both_losses:
            if (
                t_reset_idcs is None and t_stop_idx is None
            ) and not ys_from_grad is None:
                forw_full = ys_from_grad
            else:
                forw_full = forward_func(
                    (
                        y0
                        if t_reset_idcs is None
                        else pytree_select_time(get_y0(opt_state_y0), 0)
                    ),
                    t_evals,
                    get_params(opt_state),
                    get_iparams(opt_state_ip),
                    exparams,
                )
            loss = loss_func(
                forw_full,
                get_params(opt_state),
                get_iparams(opt_state_ip),
                exparams,
                ys_target,
            )

            if len(loss) > 5:
                loss_print = [jnp.mean(loss)]
            else:
                loss_print = loss
            loss_string = "Loss: " + ("{:.1e}, " * len(loss_print)).format(*loss_print)
        act_loss_string = ""
        if not (t_reset_idcs is None and t_stop_idx is None):
            if ys_from_grad is None:
                y0_here = get_y0(opt_state_y0)
                if t_reset_idcs is None:
                    y0_here = tree_util.tree_map(lambda x: x[:, jnp.newaxis], y0_here)
                forw_act = resetted_forward_func(
                    y0_here,
                    t_evals,
                    get_params(opt_state),
                    get_iparams(opt_state_ip),
                    exparams,
                    ys_target,
                    t_stop_idx,
                    t_reset_idcs,
                )
            else:
                forw_act = ys_from_grad
            act_loss = loss_func(
                forw_act,
                get_params(opt_state),
                get_iparams(opt_state_ip),
                exparams,
                ys_target,
            )
            if len(act_loss) > 5:
                act_loss_print = [jnp.mean(act_loss)]
            else:
                act_loss_print = act_loss
            act_loss_string = "Stopped/Resetted Loss: " + (
                "{:.1e}, " * len(act_loss_print)
            ).format(*act_loss_print)
            loss = (
                act_loss  # If actloss is computed it is returned as the relevant loss
            )
        return loss, loss_string, act_loss_string

    for step in range(epochs):
        grads, grads_ip, grads_y0, ys_forward = grad_func(
            get_y0(opt_state_y0),
            t_evals,
            get_params(opt_state),
            get_iparams(opt_state_ip),
            exparams,
            ys_target,
            t_stop_idx=t_stop_idx,
            time_decay=time_decay,
            t_reset_idcs=t_reset_idcs,
        )

        loss, loss_string, act_loss_string = losses_for_print(
            opt_state, opt_state_ip, opt_state_y0, ys_from_grad=ys_forward
        )
        if step % save_interval == 0:
            losses.append(loss)
            errors.append(
                {
                    "params": pytree_norm(params_true, get_params(opt_state)),
                    "iparams": pytree_norm(iparams_true, get_iparams(opt_state_ip)),
                    "y0": pytree_norm(dataset.y0, get_y0(opt_state_y0)),
                }
            )
            params_history.append(
                {
                    "params": to_np_arr(get_params(opt_state)),
                    "iparams": to_np_arr(get_iparams(opt_state_ip)),
                    "y0": to_np_arr(get_y0(opt_state_y0)),
                }
            )
        if (not np.isnan(print_interval)) and (
            step % print_interval == 0 or step == epochs - 1
        ):
            print(
                "Epoch {:03d}: ".format(step),
                loss_string,
                "Params Err.: {:.1e}, y0 error: {:.1e}, "
                "Params Norm: {:.1e}, iParams Err.: {:.1e}, iParams Norm: {:.1e},".format(
                    pytree_norm(params_true, get_params(opt_state)),
                    pytree_norm(dataset.y0, get_y0(opt_state_y0)),
                    pytree_norm(get_params(opt_state)),
                    pytree_norm(iparams_true, get_iparams(opt_state_ip)),
                    pytree_norm(get_iparams(opt_state_ip)),
                ),
                act_loss_string,
            )
        if any(
            [
                x.any()
                for x in tree_util.tree_flatten(
                    tree_util.tree_map(jnp.isnan, (grads, grads_ip, grads_y0))
                )[0]
            ]
        ):
            raise Exception(
                "Gradients resulted to nans. Maybe try the back_check function "
                "to see is your backward pass is instable. In that case it can help "
                "to increase the number of Backups ('N_backups') used in between "
                "time points."
            )
        opt_state = opt_update(step, grads, opt_state)
        opt_state_y0 = opt_update_y0(step, grads_y0, opt_state_y0)
        opt_state_ip = opt_update_ip(step, grads_ip, opt_state_ip)
        dataset.params_train = get_params(opt_state)
        dataset.iparams_train = get_iparams(opt_state_ip)
        dataset.y0_train = get_y0(opt_state_y0)
        if (not kwargs_adoptODE["terminal_loss"] is None) and loss < kwargs_adoptODE[
            "terminal_loss"
        ]:
            break
    dataset.params_save = params_history
    dataset.update_sol()
    dataset.final_gradients = {
        "params": to_np_arr(grads),
        "iparams": to_np_arr(grads_ip),
        "y0": to_np_arr(grads_y0),
    }
    return (
        {
            "params": get_params(opt_state),
            "iparams": get_iparams(opt_state_ip),
            "y0": get_y0(opt_state_y0),
        },
        losses,
        errors,
        params_history,
    )


def _check_kwargs_adoptODE(kwargs_adoptODE, t_evals):
    """Checking the kwargs_adoptODE dictionary for errors and filling defaults."""
    kw_adoptODE_necessary_keys = ["lr", "epochs"]
    if not all(key in kwargs_adoptODE for key in kw_adoptODE_necessary_keys):
        raise Exception(
            "One of the necessary keys was not included in kwargs_adoptODE. The "
            "necessary keys are: 'lr': learning rate, 'epochs': Number "
            "of trainig epochs"
        )
    kw_adoptODE_defaults = {
        "terminal_loss": None,
        "lr_y0": 0.0,
        "lr_ip": 0.0,
        "N_backups": 1,
        "atol": 1e-5,
        "rtol": 1e-5,
        "t_stop_idx": None,
        "time_decay": 1.0,
        "t_reset_idcs": None,
        "lr_decay": None,
        "lr_decay_y0": None,
        "lr_decay_ip": None,
        "EOM_dt_dependent": False,
        "vectorize_t_evals": False,
        "optimizer": OptBounded.adam_bounded,
        "optimizer_ip": OptBounded.adam_bounded,
        "optimizer_y0": OptBounded.adam_bounded,
        "optimizer_kwargs": {"b1": 0.7, "b2": 0.8},
        "optimizer_ip_kwargs": {"b1": 0.7, "b2": 0.8},
        "optimizer_y0_kwargs": {"b1": 0.7, "b2": 0.8},
        "upper_b": None,
        "lower_b": None,
        "upper_b_ip": None,
        "lower_b_ip": None,
        "upper_b_y0": None,
        "lower_b_y0": None,
        "multi_measurement_constraint": None,
        "custom_scheduel": None,
        "custom_scheduel_ip": None,
        "custom_scheduel_y0": None,
    }
    for k in kw_adoptODE_defaults.keys():
        if not k in kwargs_adoptODE:
            kwargs_adoptODE[k] = kw_adoptODE_defaults[k]
    kw_adoptODE_optional_keys = ["ODE_solver", "dt"]
    if (
        kwargs_adoptODE["EOM_dt_dependent"] is True
        and not "ODE_solver" in kwargs_adoptODE
    ):
        raise Exception(
            "dt dependent EOM is only supported if a fitting ODE_solver is "
            "provided to kwargs_adoptODE via the 'ODE_solver' keyword."
        )
    unrecognised_keys = [
        key
        for key in kwargs_adoptODE
        if key
        not in kw_adoptODE_necessary_keys
        + list(kw_adoptODE_defaults.keys())
        + kw_adoptODE_optional_keys
    ]
    if len(unrecognised_keys) > 0:
        raise Exception(
            "Unrecognised keys were passed via kwargs_adoptODE: {}\n Are you looking "
            "for one of the following? {}: ".format(
                str(unrecognised_keys),
                str(
                    kw_adoptODE_necessary_keys
                    + list(kw_adoptODE_defaults.keys())
                    + kw_adoptODE_optional_keys
                ),
            )
        )
    if (
        not kwargs_adoptODE["t_reset_idcs"] is None
        and not isinstance(kwargs_adoptODE["t_reset_idcs"], tuple)
        and not isinstance(kwargs_adoptODE["t_reset_idcs"], int)
    ):
        raise Exception("'t_stop_idcs' must be a tuple of ints, no array or list!")
    if (
        not (
            kwargs_adoptODE["t_reset_idcs"] is None
            or kwargs_adoptODE["t_stop_idx"] is None
        )
        and kwargs_adoptODE["t_reset_idcs"][-1] >= kwargs_adoptODE["t_stop_idx"]
    ):
        raise Exception(
            "The last index in 't_reset_idcs' has to be smaller"
            "than the 't_stop_idx'."
        )
    elif (
        (not kwargs_adoptODE["t_reset_idcs"] is None)
        and kwargs_adoptODE["t_stop_idx"] is None
        and kwargs_adoptODE["t_reset_idcs"][-1] >= t_evals.shape[-1] - 1
    ):
        raise Exception(
            "The last index in 't_reset_idcs' has to be smaller "
            "than number of times in t_evals - 1."
        )
    return kwargs_adoptODE


class dataset_adoptODE:
    """Combining a defined system with data to be fitted in one object.

    This class gives the main object to be instanciated for every problem to be
    solved. It combines the general information of the system as pass via the
    define_system function as well as data to be fitted and the adoptODE framework
    implemented above. It also serves as a container to store the functions
    compiled for a certai problem, such that for some changes recompilation can
    be avoided.
    Additionally the parameters are saved in this class. Three different types
    of parameters are distinguished:

    1. params:

        The normal general parameters of the system which are unknown.

    2. iparams:

        Also unknown parameters, which might depend on the
        realization, e.g. if for example multiple realizations are
        considered simultaneously, params are one set of parameters
        to fit all of them, while iparams, short for individual
        parameters, are seperatly fitted for each realization.
        While within the EOM iparams is always for a single system,
        within the PhysadoptODEDataclass iparams has a leading axis and
        contains the values for every system in the batch

    3. exparams:

        Parameters which are known, but cannot be build into the
        systems definition because they might be different between
        different observations. While within the EOM exparams is
        always for a single system, within the PhysadoptODEDataclass
        iparams has a leading axis and contains the values for every
        system in the batch.

    Attributes:
        kwargs_adoptODE: Dictionary of keyword settings for the adoptODE, described below.
        def_sys_func: The function defining the system, see define_system_dummy().
        ys: The data points to be fitted, with a leading axis for the different
            systems in the batch. A leading axis of rank one is required for a
            single system. The second axis is different time points.
        t_evals: An array of the times at which the system is evaluated to obtain ys. If
            different times are desired for each system in the batch, this can be a
            two dimensional array, first axis giving the different system. For more
            details see 'vectorize_t_evals' keyword in kwargs_adoptODE.
        N_sys: The number of different systems fittet simultaneously.
        kwargs_sys: The kwargs passed to the define_system function to specify the system.
        eom: The Equation of Motion of the system.
        loss: The loss function to be minimized.
        gen_params: The function generating the system params,
            returning params, iparams, exparams
        gen_y0: Function generating initial conditions for a single system.
        add_functions: Possible helper functions used by def_sys_func to construct for example
            the EOM can be returned by def_sys_func in this dictionary for later use.
        y0: The initial conditions for the batch of the system as provided to the
            class. This can be the earliest entry of ys. If provided via the true_y0
            keyword, this is changed to the true y0 and used to calculate deviation
            of the learned y0 from the true solution. The first axis counts different
            system copies, in case of 't_reset_idcs' or 't_stop_idx' used, a second
            axis with lenght of reset indices or one if only 't_stop_idx' is used has
            to be passed.
        y0_train: The y0 value actually used during training. If the learning rate for y0
            is non-zero, this is manipulated during training to improve agreement
            with data. Has the same shape as descriped for y0.
        params_train: The params values used and manipulated during training.
        iparams_train: The iparams values used and manipulated during training.
        exparams: The provided, known exparams values used during training.
        params: If provided via 'true_params' used to compare the learned params. This is
            for example the case for :class:`dataset_adoptODE`s generated via the simple
            simulation function. Otherwise this is NaNs with the same shape as
            params_train.
        iparams: If provided via 'true_iparams' used to compare the learned iparams.
            Otherwise this is NaNs with the same shape as iparams_train.
        final_gradients: After a training run, the last values of the gradients are saved in this
            attribute as a dictionary, with the keys 'params', 'iparams' and 'y0'.
        forward: Function solving a batch of systems forward.
        grad_ad: Function using the adjoint method to compute gradients for a batch of
            systems.
        v_loss: Vectorized version of the loss function, computing the loss for a
            batch of systems.
        params_save: A list of the parameters at different epochs saved during training.
        ys_sol: The solution as computed with the current learned values for parameters
            and initial conditions.

    Keywords controlling adoptODE (:code:`kwargs_adoptODE`):
        'epochs':
            Number of epochs to run training for. (NECESSARY)
        'lr':
            Learning rate for params. (NECESSARY)
        'lr_y0':
            Learning rate for initial conditions. Default: 0.
        'lr_ip':
            Learning rate for iparams. Default: 0.
        'lr_decay':
            Basis for exponential decay of learning rate with epochs. Default: None.
        'lr_decay_y0':
            Basis for exponential decay of learning rate for initial conditions y0
            with epochs. Default: None.
        'lr_decay_ip':
            Basis for exponential decay of learning rate for individual parameters
            with epochs. Default: None.
        'N_backups':
            Number of subdivisions fo each interval between recorded times at which
            backups of the system state from a forward pass are used while
            propagating backwards. Increasing can help to fix issues with stability
            in the system state solving in negative time direction. Default: 1.
        'atol':
            Absolute tolerance of ODE solver. Default: 1e-5.
        'rtol':
            Relative tolerance of ODE solver. Default: 1e-5.
        'ODE_solver':
            If passed, this solver is used instead of the Dopri5 mixed 4th/5th order
            Runge-Kutter integrator from jax.experimental.ode. Should mimic the
            syntax. Default: None.
        'dt':
            Fixed stepsize, if a fixed stepsize solver should be used. A fitting
            solver has to be passe via 'ODE_solver' which takes dt as 4th argument.
            Default: None.
        'EOM_dt_dependent':
            For certain systems it might be usefull to use the stepsize within the
            EOM. This specifies this behaviour. If used, the EOM takes dt as 3rd
            argument. The used solver has to be constructed such that it also passes
            dt as third argument to the EOM while valuating. Default: False.
        't_stop_idx':
            If not the whole time-domain should be used to compute gradients, but
            just the first timesteps, this can be used, indicating the index of the
            time in t_evals at which to stop. Changings this does NOT require running
            :func:`update_equations` and recompiling. If used, the same code as for
            't_reset_idcs' is used, making it necessary to submit the initial
            conditions y0 with an additional axis behind the axis for different
            systems.
        'time_decay':
            Modulates importance of later times compared to early times, by making
            gradient contributions and the adjoint state decay exponentially with
            time with 'time_decay' as basis. Hence if chosen <1, contributions
            further from the initial conditions are reduced.
        't_reset_idcs':
            Instead of choosing initial conditions once and propagating from there,
            this allows to reload initial conditions at multiple indices indicating
            the times in t_evals at which to reset the system state to the target
            state at that time. Hence the time domain is splitted in multiple small
            pieces, each fitted from its beginning. Has to be passed as a tuple of
            integers. If used, the initial conditions y0 have to have an additional
            axes for the different reset times, as second axes after the axis for
            different systems.
        'vectorize_t_evals':
            Usually the times of evaluations are the same for all copies of a system
            within a batch. If different copies are observed at different times, this
            keyword can be used to pass a 2D array for t_evals, with the leading axis
            differentiating the different copies of systems. As this has to be an
            array each copy still has to be observed equally often. If this is not
            desired one can choose loss function and target values ys_target
            appropriately, such that certain oberservations for certain system copies
            are made not to contribute.
        'optimizer', 'optimizer_ip', 'optimizer_y0':
            Optimizer to be used for optimization on params, iparams and y0
            respectively. Default is OptBounded.adam_bounded.
        'optimizer_kwargs', optimizer_ip_kwargs', optimizer_y0_kwargs':
            Keyword arguments supplying hyperparameters to the selected optimizer.
            Defauls is b1=0.7 and b2=0.8, as hyperparameters for the adam optimizer
            selected by default.
        'upper_b','lower_b','upper_b_ip','lower_b_ip','upper_b_y0','lower_b_y0':
            Boundaries for the parameter search, upper and lower for each type of
            parameters. Can be either a number, applied to all parameters, or a
            pytree of same or broadcastable shape as the respective parameters,
            setting individual boundaries.
        'multi_measurement_constraint':
            Loss function for iParams, controlling how iParams of different
            measurements behave with respect to each other. For example, each
            observation is fitted an individual offset in time, but the some of
            offsets should be zero: Include here the square of the summed offsets.
        'custom_scheduel', 'custom_scheduel_ip', 'custom_scheduel_y0':
            Learning rate scheduels for the different optimizations, taking the
            current epoch as input and returning the learning rate to be used. If
            specified, the learning rate 'lr' is rendered obsolete.

    """

    def __init__(
        self,
        def_sys_func,
        ys,
        t_evals,
        kwargs_sys,
        kwargs_adoptODE,
        exparams=None,
        true_params=None,
        true_iparams=None,
        true_y0=None,
        params_train=None,
        iparams_train=None,
        y0_train=None,
    ):
        """Initialization

        Args:
          def_sys_func:
            The function defining the system, see define_system_dummy().
          ys:
            The data points to be fitted, with a leading axis for the different
            systems in the batch. A leading axis of rank one is required for a
            single system. The second axis is different time points.
          t_evals:
            An array of the times at which the system is evaluated to obtain ys. If
            different times are desired for each system in the batch, this can be a
            two dimensional array, first axis giving the different system. For more
            details see 'vectorize_t_evals' keyword in kwargs_adoptODE.
          kwargs_sys:
            The kwargs passed to the define_system function to specify the system.
          kwargs_adoptODE:
            Dictionary of keyword settings for the adoptODE, see class description.
          true_y0:
            True initial conditions to compare against.
          params_train:
            Initial guess of params for training.
          iparams_train:
            Initial guess of iparams for training.
          exparams
            The provided, known exparams values used during training.
          true_params:
            True params to compare against.
          true_iparams:
            True iparams to compare against.
          y0_train:
            Initial guess for initial conditions.
        """

        self.kwargs_adoptODE = _check_kwargs_adoptODE(kwargs_adoptODE, t_evals)
        self.def_sys_func = def_sys_func
        self.ys = ys
        self.t_evals = t_evals
        self.N_sys = tree_util.tree_leaves(ys)[0].shape[0]
        self.kwargs_sys = kwargs_sys
        self.f, self.sng_loss, self.gen_params, self.gen_y0, self.add_functions = (
            def_sys_func(**self.kwargs_sys)
        )

        if y0_train is None:
            if kwargs_adoptODE["t_reset_idcs"] is None:
                self.y0_train = tree_util.tree_map(lambda x: np.copy(x)[:, 0], self.ys)
            else:
                self.y0_train = tree_util.tree_map(
                    lambda x: np.copy(x)[:, np.array(kwargs_adoptODE["t_reset_idcs"])],
                    self.ys,
                )
        else:
            if not kwargs_adoptODE["t_reset_idcs"] is None and y0_train.shape[1] != len(
                kwargs_adoptODE["t_reset_idcs"]
            ):
                raise Exception(
                    "If 't_reset_idcs' is used, the provided y0_train "
                    "has to have a second axis giving the initial condition for "
                    "each time the system is reset."
                )
            self.y0_train = y0_train

        self.params_train, self.iparams_train, self.exparams = self.gen_params()
        if not exparams is None:
            self.exparams = exparams
        if not params_train is None:
            self.params_train = params_train
        if not iparams_train is None:
            self.iparams_train = iparams_train
        if true_params is None:
            self.params = tree_util.tree_map(lambda x: x * jnp.nan, self.params_train)
        else:
            self.params = true_params
        if true_iparams is None:
            self.iparams = tree_util.tree_map(lambda x: x * jnp.nan, self.iparams_train)
        else:
            self.iparams = true_iparams
        if not true_y0 is None:
            self.y0 = true_y0
        else:
            self.y0 = tree_util.tree_map(lambda x: x * np.nan, self.y0_train)
        self.equations = EquationsNODE(self.f, self.sng_loss, **self.kwargs_adoptODE)
        self.forward, self.gradient, self.loss, self.resetted_forward = (
            self.equations.forward,
            self.equations.gradient,
            self.equations.loss,
            self.equations.resetted_forward,
        )
        self.params_save = None
        self.ys_sol = None
        self.final_gradients = None

    def update_sol(self):
        """Update ys_sol with the current values of parameters and initial
        conditions. Automatically called at the end of training."""
        if (self.kwargs_adoptODE["t_reset_idcs"] is None) and (
            self.kwargs_adoptODE["t_stop_idx"] is None
        ):
            self.ys_sol = tree_util.tree_map(
                np.array,
                self.forward(
                    self.y0_train,
                    self.t_evals,
                    self.params_train,
                    self.iparams_train,
                    self.exparams,
                ),
            )
        else:
            t_reset_idcs_here = (
                (0,)
                if self.kwargs_adoptODE["t_reset_idcs"] is None
                else self.kwargs_adoptODE["t_reset_idcs"]
            )
            self.ys_sol = tree_util.tree_map(
                np.array,
                self.resetted_forward(
                    self.y0_train,
                    self.t_evals,
                    self.params_train,
                    self.iparams_train,
                    self.exparams,
                    self.ys,
                    self.kwargs_adoptODE["t_stop_idx"],
                    t_reset_idcs_here,
                ),
            )

    def update_equations(self):
        """Changes in the system defining parameters or fixed kwargs_adoptODE keywords
        (e.g. the solver or the tolerances) requires to rebuiled and recompile
        the main equation, which is executed by this function. Works only if the
        shape of parameters and system state remains unchanged, otherwise a new
        dataset_adoptODE class has to be initialized.
        """
        self.kwargs_adoptODE = _check_kwargs_adoptODE(
            self.kwargs_adoptODE, self.t_evals
        )
        self.f, self.sng_loss, self.gen_params, self.gen_y0, self.add_Functions = (
            self.def_sys_func(**self.kwargs_sys)
        )
        self.equations = EquationsNODE(self.f, self.sng_loss, **self.kwargs_adoptODE)
        self.forward, self.gradient, self.loss, self.resetted_forward = (
            self.equations.forward,
            self.equations.gradient,
            self.equations.loss,
            self.equations.resetted_forward,
        )

    def backwards_check(
        self, times, y0=None, params=None, iparams=None, exparams=None, system_to_use=0
    ):
        """Providing forwad and backward pass results to check stability.

        Forward and backward propagation with given parameters for a single system.

        Args:
            Standard arguments as listed above.
            times: Here different: arbitrary timepoints to evaluate the system at.

        Returns:
            ys_forward: Results of the forward pass, evaluated at times.
            ys_backward: Resuts of the backward pass, evaluated at times.
        """

        def sgc(pytree):  # select given copy of system
            return tree_util.tree_map(lambda x: x[system_to_use], pytree)

        y0 = sgc(self.y0_train) if y0 is None else sgc(y0)
        y0 = (
            y0
            if self.kwargs_adoptODE["t_reset_idcs"] is None
            else pytree_select_system(y0, 0)
        )
        params = self.params_train if params is None else params
        iparams = sgc(self.iparams_train) if iparams is None else sgc(iparams)
        exparams = sgc(self.exparams) if exparams is None else sgc(exparams)

        ys_forward = self.equations._forward(y0, times, params, iparams, exparams)
        a = tree_util.tree_map(jnp.zeros_like, y0)
        s0 = (
            tree_util.tree_map(lambda x: x[-1], ys_forward),
            a,
            tree_util.tree_map(lambda x: 0 * x, params),
            tree_util.tree_map(lambda x: 0 * x, iparams),
        )
        t_evals_rev = times[-1] - jnp.flip(times)
        sol = self.equations._odeint_back(
            self.equations._aug_dyn,
            s0,
            t_evals_rev,
            params,
            iparams,
            times[-1],
            exparams,
            rtol=self.kwargs_adoptODE["rtol"],
            atol=self.kwargs_adoptODE["atol"],
        )
        ys_backward = tree_util.tree_map(lambda x: jnp.flip(x, axis=0), sol[0])
        return ys_forward, ys_backward

    def gradient_check(
        self,
        t_indices,
        y0=None,
        t_evals=None,
        params=None,
        iparams=None,
        exparams=None,
        ys_target=None,
        detailed_times=None,
        system_to_use=0,
    ):
        """Providing forwad, backward pass and gradients to check stability.

        Forward and backward propagation with given parameters for a single system.
        In addition to back_check also gradients are returned

        Args:
          Standard arguments as listed above.
          t_indices: The indices of times in t_evals to be used. Should start at 0.
          detailed_times: If you dont want to go back from on time in t_evals to
                          another, pass an array of times here. The backward pass
                          then happens from t_evals[t_indices[-1]] going backwards
                          by the amount of times indicated by the array passed.

        Returns:
          ys_forward: Results of the forward pass, evaluated at t_evals.
          ys_backward: Resuts of the backward pass, evaluated at t_evals.
          gradients: The gradients accumulated in the backward pass.
        """

        def sgc(pytree):  # select given copy of system
            return tree_util.tree_map(lambda x: x[system_to_use], pytree)

        y0 = sgc(self.y0_train) if y0 is None else sgc(y0)
        y0 = (
            y0
            if self.kwargs_adoptODE["t_reset_idcs"] is None
            else pytree_select_system(y0, 0)
        )
        t_evals = self.t_evals if t_evals is None else t_evals
        params = self.params_train if params is None else params
        iparams = sgc(self.iparams_train) if iparams is None else sgc(iparams)
        exparams = sgc(self.exparams) if exparams is None else sgc(exparams)
        ys_target = sgc(self.ys) if ys_target is None else sgc(ys_target)

        t_evals_here = jnp.array([t_evals[i] for i in t_indices])
        ys_forward = self.equations._forward(y0, t_evals, params, iparams, exparams)
        a = self.equations._dloss_dy(
            ys_forward, t_indices[-1], params, iparams, exparams, ys_target
        )
        s0 = (
            tree_util.tree_map(lambda x: x[t_indices[-1]], ys_forward),
            a,
            tree_util.tree_map(lambda x: 0 * x, params),
            tree_util.tree_map(lambda x: 0 * x, iparams),
        )
        if detailed_times is None:
            t_evals_rev = t_evals_here[-1] - jnp.flip(t_evals_here)
        else:
            t_evals_rev = t_evals_here[-1] - jnp.flip(detailed_times)
        sol = self.equations._odeint_back(
            self.equations._aug_dyn,
            s0,
            t_evals_rev,
            params,
            iparams,
            t_evals_here[-1],
            exparams,
            rtol=self.kwargs_adoptODE["rtol"],
            atol=self.kwargs_adoptODE["atol"],
        )
        ys_backward = tree_util.tree_map(lambda x: jnp.flip(x, axis=0), sol[0])
        gradients = tree_util.tree_map(lambda x: jnp.flip(x, axis=0), sol[1:])
        return ys_forward, ys_backward, gradients

    def nan_locator(self):
        """Function to support tracking down NaN errors.

        Takes different substeps of the framework, and returns if NaNs where
        encounterd.

        Returns:
          Dictionary of the following intermediate results used by the framework:
            'Data': Training Data,
            'Forward': Evolution computed in forward pass,
            'dloss_dp': Jacobian of Loss w.r.t. params,
            'dloss_dip': Jacobian of Loss w.r.t. iparams,
            'dloss_dy': Jacobian of Loss w.r.t. system state,
            'Grads params': Gradient of params,
            'Grads iparams': Gradient of iparams,
            'Grads y0': Gradient of initial conditions."""
        ys_data = self.ys
        grad_p, grad_ip, grad_y0, ys_forw = self.gradient(
            self.y0_train,
            self.t_evals,
            self.params_train,
            self.iparams_train,
            self.exparams,
            ys_data,
            t_stop_idx=self.kwargs_adoptODE["t_stop_idx"],
            time_decay=self.kwargs_adoptODE["time_decay"],
            t_reset_idcs=self.kwargs_adoptODE["t_reset_idcs"],
        )

        ys_forw_single = pytree_select_system(ys_forw, 0)
        ys_data_single = pytree_select_system(ys_data, 0)
        if not self.kwargs_adoptODE["t_reset_idcs"] is None:
            ys_forw_single = pytree_select_system(
                ys_forw_single, self.kwargs_adoptODE["t_reset_idcs"][0]
            )
            ys_data_single = pytree_select_system(
                ys_data_single, self.kwargs_adoptODE["t_reset_idcs"][0]
            )
        iparams_single = pytree_select_system(self.iparams_train, 0)
        exparams_single = pytree_select_system(self.exparams, 0)

        dloss_dp = self.equations._dloss_dp(
            ys_forw_single,
            self.params_train,
            iparams_single,
            exparams_single,
            ys_data_single,
        )
        dloss_dip = self.equations._dloss_dip(
            ys_forw_single,
            self.params_train,
            iparams_single,
            exparams_single,
            ys_data_single,
        )
        dloss_dy = self.equations._dloss_dy(
            ys_forw_single,
            jnp.arange(len(self.t_evals)),
            self.params_train,
            iparams_single,
            exparams_single,
            ys_data_single,
        )

        def any_nan_tree(x):
            return jnp.isnan(flatten_util.ravel_pytree(x)[0]).any()

        results = {
            "Data": ys_data,
            "Forward": ys_forw,
            "dloss_dp": dloss_dp,
            "dloss_dip": dloss_dip,
            "dloss_dy": dloss_dy,
            "Grads params": grad_p,
            "Grads iparams": grad_ip,
            "Grads y0": grad_y0,
        }
        for key in results.keys():
            print(key + " has NaNs: ", any_nan_tree(results[key]))
        return results


def simple_simulation(
    def_sys_func,
    t_evals,
    kwargs_sys,
    kwargs_adoptODE,
    rel_noise=0.0,
    abs_noise=0.0,
    params=None,
    iparams=None,
    exparams=None,
    y0=None,
    params_train=None,
    iparams_train=None,
    y0_train=None,
):
    """Setting up a dataset_adoptODE with data generated by simulation.

    For test purposes and to check whether obtained results can be trusted to
    reliable reproduce true parameters of the system, after a system is defined
    obervations can be simulated and afterwards the adoptODE framework applied to
    recover the parameters used for simulation. This method performs the data
    generation and returns a PhysadoptODEData class. Params, iparamd and exparams
    and initial conditions can either be passed via respective keywords, or
    otherwise are generated with the gen_params() and gen_y0() functions returned
    by the define_system function.

    Args:
      def_sys_func:
        The function defining the system, see define_system_dummy().
      t_evals:
        An array of the times at which the system is evaluated to obtain ys. If
        different times are desired for each system in the batch, this can be a
        two dimensional array, first axis giving the different system. For more
        details see 'vectorize_t_evals' keyword in kwargs_adoptODE.
      kwargs_sys:
        The kwargs passed to the define_system function to specify the system.
      kwargs_adoptODE:
        Dictionary of keyword settings for the adoptODE, see class description.
      rel_noise:
        Noise amplitude applied to generated data. Noise is scaled relative to
        the mean standard deviation, computed for each PyTree leaf.
      abs_noise:
        Noise amplitude applied to generated data. Anabsoulte noise with this
        stdv. is applied.
      params:
        If passed params used to generate data.
      iparams:
        If passed iparams used to generate data.
      exparams:
        If passed exparams used to generate data.
      y0:
        If passed initial conditions used to generate data.
      params_train:
        Initial guess of params for training.
      iparams_train:
        Initial guess of iparams for training.
      y0_train:
        Initial guess for initial conditions.

    Returns:
      dataset_adoptODE with generated data.
    """
    kwargs_adoptODE = _check_kwargs_adoptODE(kwargs_adoptODE, t_evals)
    if not "N_sys" in kwargs_sys:
        raise Exception(
            "For simulation a key 'N_sys' passing the number of "
            "systems to be generated is necessary in kwargs_sys!"
        )
    n_sys = kwargs_sys["N_sys"]
    f, loss, gen_params, gen_y0, _ = def_sys_func(**kwargs_sys)
    equationsNODE = EquationsNODE(f, loss, **kwargs_adoptODE)
    forward = equationsNODE.forward
    lparams, liparams, lexparams = gen_params()
    if not params is None:
        lparams = tree_util.tree_map(lambda x: x, params)
    if not iparams is None:
        liparams = tree_util.tree_map(lambda x: x, iparams)
    if not exparams is None:
        lexparams = exparams
    if y0 is None:
        ly0_list = [gen_y0() for i in range(n_sys)]
        ly0 = tree_util.tree_map(lambda x: np.expand_dims(x, axis=0), ly0_list[0])
        for ly0_here in ly0_list[1:]:
            ly0 = tree_util.tree_map(
                lambda x, y: np.concatenate((x, np.expand_dims(y, axis=0))),
                ly0,
                ly0_here,
            )
    else:
        ly0 = tree_util.tree_map(np.array, y0)
    lys = tree_util.tree_map(
        np.array, forward(ly0, t_evals, lparams, liparams, lexparams)
    )
    if not kwargs_adoptODE["t_reset_idcs"] is None:
        ly0 = tree_util.tree_map(
            lambda x: np.copy(x)[:, np.array(kwargs_adoptODE["t_reset_idcs"])], lys
        )
    lys = tree_util.tree_map(
        lambda x: x
        + (rel_noise * np.mean(np.std(x, axis=1)) + abs_noise)
        * np.random.normal(size=x.size).reshape(x.shape),
        lys,
    )

    return dataset_adoptODE(
        def_sys_func,
        lys,
        t_evals,
        kwargs_sys,
        kwargs_adoptODE,
        exparams=lexparams,
        true_params=lparams,
        true_iparams=liparams,
        true_y0=ly0,
        params_train=params_train,
        iparams_train=iparams_train,
        y0_train=y0_train,
    )
