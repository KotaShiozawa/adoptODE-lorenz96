"""Adaption of the JAX included ODE solver, such that the stepsize dt
used at the current step is passed to the equation of motion, which
therefore func(y, t, dt, *args) instead func(y, t, *args)."""

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JAX-based Dormand-Prince ODE integration with adaptive stepsize.
Integrate systems of ordinary differential equations (ODEs) using the JAX
autograd/diff library and the Dormand-Prince method for adaptive integration
stepsize calculation. Provides improved integration accuracy over fixed
stepsize integration methods.
For details of the mixed 4th/5th order Runge-Kutta integration method, see
https://doi.org/10.1090/S0025-5718-1986-0815836-3
Adjoint algorithm based on Appendix C of https://arxiv.org/pdf/1806.07366.pdf
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import core, lax
from jax._src.util import safe_map, safe_zip
try:
    from jax.extend import linear_util as lu
except:
    from jax import linear_util as lu
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves

map = safe_map
zip = safe_zip
# first_guess_dt = 1e-2


def _ravel_first_arg(f, unravel):
    return _ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped


@lu.transformation
def _ravel_first_arg_(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


def _interp_fit_dopri(y0, y1, k, dt):
    # Fit a polynomial to the results of a Runge-Kutta step.
    dps_c_mid = jnp.array(
        [
            6025192743 / 30085553152 / 2,
            0,
            51252292925 / 65400821598 / 2,
            -2691868925 / 45128329728 / 2,
            187940372067 / 1594534317056 / 2,
            -1776094331 / 19743644256 / 2,
            11237099 / 235043384 / 2,
        ],
        dtype=y0.dtype,
    )
    y_mid = y0 + dt * jnp.dot(dps_c_mid, k)
    return jnp.asarray(_fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))


def _fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
    a = -2.0 * dt * dy0 + 2.0 * dt * dy1 - 8.0 * y0 - 8.0 * y1 + 16.0 * y_mid
    b = 5.0 * dt * dy0 - 3.0 * dt * dy1 + 18.0 * y0 + 14.0 * y1 - 32.0 * y_mid
    c = -4.0 * dt * dy0 + dt * dy1 - 11.0 * y0 - 5.0 * y1 + 16.0 * y_mid
    d = dt * dy0
    e = y0
    return a, b, c, d, e


def _runge_kutta_step(func, y0, f0, t0, dt):
    # Dopri5 Butcher tableaux
    alpha = jnp.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0, 0], dtype=f0.dtype)
    beta = jnp.array(
        [
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ],
        dtype=f0.dtype,
    )
    c_sol = jnp.array(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=f0.dtype
    )
    c_error = jnp.array(
        [
            35 / 384 - 1951 / 21600,
            0,
            500 / 1113 - 22642 / 50085,
            125 / 192 - 451 / 720,
            -2187 / 6784 - -12231 / 42400,
            11 / 84 - 649 / 6300,
            -1.0 / 60.0,
        ],
        dtype=f0.dtype,
    )

    def body_fun(i, k):
        ti = t0 + dt * alpha[i - 1]
        yi = y0 + dt * jnp.dot(beta[i - 1, :], k)
        ft = func(yi, ti, dt)
        return k.at[i, :].set(ft)

    k = jnp.zeros((7, f0.shape[0]), f0.dtype).at[0, :].set(f0)
    k = lax.fori_loop(1, 7, body_fun, k)

    y1 = dt * jnp.dot(c_sol, k) + y0
    y1_error = dt * jnp.dot(c_error, k)
    f1 = k[-1]
    return y1, f1, y1_error, k


def _abs2(x):
    if jnp.iscomplexobj(x):
        return x.real**2 + x.imag**2
    else:
        return x**2


def odeint(func, y0, t, dt, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf):
    """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

    Args:
        func: function to evaluate the time derivative of the solution `y` at time
            `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
        y0: array or pytree of arrays representing the initial value for the state.
        t: array of float times for evaluation, like `jnp.linspace(0., 10., 101)`,
            in which the values must be strictly increasing.
        *args: tuple of additional arguments for `func`, which must be arrays
            scalars, or (nested) standard Python containers (tuples, lists, dicts,
            namedtuples, i.e. pytrees) of those types.
        rtol: float, relative local error tolerance for solver (optional).
        atol: float, absolute local error tolerance for solver (optional).
        mxstep: int, maximum number of steps to take for each timepoint (optional).

    Returns:
        Values of the solution `y` (i.e. integrated system values) at each time
        point in `t`, represented as an array (or pytree of arrays) with the same
        shape/structure as `y0` except with a new leading axis of length `len(t)`.
    """
    for arg in tree_leaves(args):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            msg = (
                "The contents of odeint *args must be arrays or scalars, but got "
                "\n{}."
            )
            raise TypeError(msg.format(arg))
    return _odeint_wrapper(func, rtol, atol, mxstep, y0, t, jnp.abs(dt), *args)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _odeint_wrapper(func, rtol, atol, mxstep, y0, ts, dt, *args):
    y0, unravel = ravel_pytree(y0)
    func = _ravel_first_arg(func, unravel)
    out = _odeint(func, rtol, atol, mxstep, y0, ts, dt, *args)
    return jax.vmap(unravel)(out)


def _odeint(func, rtol, atol, mxstep, y0, ts, dt, *args):
    func_ = lambda y, t, dt: func(y, t, dt, *args)

    def scan_fun(carry, target_t):
        def cond_fun(state):
            i, _, _, t, dt, _, _ = state
            return (t < target_t) & (i < mxstep) & (dt > 0)

        def body_fun(state):
            i, y, f, t, dt, last_t, interp_coeff = state
            next_y, next_f, next_y_error, k = _runge_kutta_step(func_, y, f, t, dt)
            next_t = t + dt
            # error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
            new_interp_coeff = _interp_fit_dopri(y, next_y, k, dt)
            # dt = optimal_step_size(dt, error_ratios)

            new = [i + 1, next_y, next_f, next_t, dt, t, new_interp_coeff]
            old = [i + 1, y, f, t, dt, last_t, interp_coeff]
            return new

        _, *carry = lax.while_loop(cond_fun, body_fun, [0] + carry)
        _, _, t, _, last_t, interp_coeff = carry
        relative_output_time = (target_t - last_t) / (t - last_t)
        y_target = jnp.polyval(interp_coeff, relative_output_time)
        return carry, y_target

    f0 = func_(y0, ts[0], dt)
    # dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
    interp_coeff = jnp.array([y0] * 5)
    init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff]
    _, ys = lax.scan(scan_fun, init_carry, ts[1:])
    return jnp.concatenate((y0[None], ys))
