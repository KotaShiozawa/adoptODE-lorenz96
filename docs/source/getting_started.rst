Getting started
===============

.. note::
    This guide is available in executable form as a Jupyter notebook at :code:`notebooks/Cookbook.ipynb`.


Define your system
------------------
Our example system is :math:`\frac{d}{dt} \text{pop} = a \cdot \text{pop} + b`, where :math:`\text{pop}` is some scalar population and :math:`a` and :math:`b` are the parameters we want to find. We assume the initial population, :math:`a`, and :math:`b` to be bounded below by zero and above by some maximum specified in :code:`kwargs_sys`.

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from jax import jit

    def define_system(**kwargs_sys):
        p_max = kwargs_sys['p_max']
        a_max = kwargs_sys['a_max']
        b_max = kwargs_sys['b_max']
        
        def gen_y0():
            ini_pop = np.random.rand()*p_max
            return {'population':ini_pop}
        
        def gen_params():
            a = np.random.rand()*a_max
            b = np.random.rand()*b_max
            return {'a':a, 'b':b}, {}, {}
            
        @jit
        def eom(y, t, params, iparams, exparams):
            pop = y['population']
            a, b = params['a'], params['b']
            return {'population':a*pop+b}

        @jit
        def loss(ys, params, iparams, 
                        exparams, targets):
            pop = ys['population']
            t_pop = targets['population']
            return jnp.mean((pop-t_pop)**2)

        return eom, loss, gen_params, gen_y0, {}

.. testsetup:: simulation

    import numpy as np
    import jax.numpy as jnp
    from jax import jit

    def define_system(**kwargs_sys):
        p_max = kwargs_sys['p_max']
        a_max = kwargs_sys['a_max']
        b_max = kwargs_sys['b_max']
        
        def gen_y0():
            ini_pop = np.random.rand()*p_max
            return {'population':ini_pop}
        
        def gen_params():
            a = np.random.rand()*a_max
            b = np.random.rand()*b_max
            return {'a':a, 'b':b}, {}, {}
            
        @jit
        def eom(y, t, params, iparams, exparams):
            pop = y['population']
            a, b = params['a'], params['b']
            return {'population':a*pop+b}

        @jit
        def loss(ys, params, iparams, 
                        exparams, targets):
            pop = ys['population']
            t_pop = targets['population']
            return jnp.mean((pop-t_pop)**2)

        return eom, loss, gen_params, gen_y0, {}
    
    from adoptODE import simple_simulation, train_adoptODE
    kwargs_sys = {
        'p_max': 2,
        'a_max': 1,
        'b_max': 3,
        'N_sys': 1
    }
    kwargs_adoptODE = {'lr':3e-2, 'epochs':200}
    t_evals = np.linspace(0,5,10)
    np.random.seed(42)
    dataset = simple_simulation(
        define_system,
        t_evals,
        kwargs_sys,
        kwargs_adoptODE
    )

The second and third dictionary of :code:`gen_params` are :code:`iparams` and :code:`exparams` we do not have in this simple example. The first two functions can be arbitrary, the :code:`eom` and :code:`loss` functions have to be implemented using the jax libraries.

Set up a simulation
-------------------
To set up a simulation we define the dictionaries :code:`kwargs\_sys` and :code:`kwargs\_NODE` as well as the times :code:`t\_evals` at which we assume to observe our system. The keyword :code:`N\_sys` gives the number of copies in terms of multi-experiment fitting, here we consider only one system.

.. code-block:: python

    from adoptODE import simple_simulation, train_adoptODE
    kwargs_sys = {
        'p_max': 2,
        'a_max': 1,
        'b_max': 3,
        'N_sys': 1
    }
    kwargs_adoptODE = {'lr':3e-2, 'epochs':200}
    t_evals = np.linspace(0,5,10)
    np.random.seed(42)
    dataset = simple_simulation(
        define_system,
        t_evals,
        kwargs_sys,
        kwargs_adoptODE
    )

In real-life applications, these simulations not only help as an easy test environment, but also to test the reliability of parameter recovery! The simulation automatically generated some parameters, and also a (wrong) initial guess for the parameter recovery, both based on the previously define :code:`gen\_params` function:

.. testcode:: simulation

    print('The true parameters used to generate the data: ', dataset.params)
    print('The inial guess of parameters for the recovery: ', dataset.params_train)

This would output

.. testoutput:: simulation
    
    The true parameters used to generate the data:  {'a': 0.3745401188473625, 'b': 2.8521429192297485}
    The inial guess of parameters for the recovery:  {'a': 0.2912291401980419, 'b': 1.8355586841671383}

Train a simulation
------------------
The easy following command trains our simulation and prints the true params in comparison to the found ones:

.. testsetup:: training

    import numpy as np
    import jax.numpy as jnp
    from jax import jit

    def define_system(**kwargs_sys):
        p_max = kwargs_sys['p_max']
        a_max = kwargs_sys['a_max']
        b_max = kwargs_sys['b_max']
        
        def gen_y0():
            ini_pop = np.random.rand()*p_max
            return {'population':ini_pop}
        
        def gen_params():
            a = np.random.rand()*a_max
            b = np.random.rand()*b_max
            return {'a':a, 'b':b}, {}, {}
            
        @jit
        def eom(y, t, params, iparams, exparams):
            pop = y['population']
            a, b = params['a'], params['b']
            return {'population':a*pop+b}

        @jit
        def loss(ys, params, iparams, 
                        exparams, targets):
            pop = ys['population']
            t_pop = targets['population']
            return jnp.mean((pop-t_pop)**2)

        return eom, loss, gen_params, gen_y0, {}

    from adoptODE import simple_simulation, train_adoptODE
    kwargs_sys = {
        'p_max': 2,
        'a_max': 1,
        'b_max': 3,
        'N_sys': 1
    }
    kwargs_adoptODE = {'lr':3e-2, 'epochs':200}
    t_evals = np.linspace(0,5,10)
    np.random.seed(42)
    dataset = simple_simulation(
        define_system,
        t_evals,
        kwargs_sys,
        kwargs_adoptODE
    )
    _ = train_adoptODE(dataset, print_interval=None)

We can now check the found parameters:

.. testcode:: training

    print('True params: ', dataset.params)
    print('Found params: ', dataset.params_train)

.. testoutput:: training

    True params:  {'a': 0.3745401188473625, 'b': 2.8521429192297485}
    Found params:  {'a': Array(0.37777752, dtype=float32), 'b': Array(2.850388, dtype=float32)}

For more accurate results, try to manipulate the learing rate or the number of epochs!

Include Data
--------------
To include data, we bring it in the same form as the shape of the state given by :code:`gen\_y0()`, but with two additional leading axes. The first counts the different experiments, and has length one here, the second runs over time points and has the same length as :code:`t\_evals`.

.. code-block:: python

    from adoptODE import dataset_adoptODE
    data = np.array(
        [ 0.86, 1.66, 2.56, 3.59, 4.75, 6.08, 7.58, 9.28, 11.21, 13.40]
    ) # Observation of population, shape (10,)
    targets = {'population':data.reshape((1,10))}
    dataset2 = dataset_adoptODE(
        define_system,
        targets,
        t_evals,
        kwargs_sys,
        kwargs_adoptODE
    )

.. testsetup:: added_data

    import numpy as np
    import jax.numpy as jnp
    from jax import jit

    def define_system(**kwargs_sys):
        p_max = kwargs_sys['p_max']
        a_max = kwargs_sys['a_max']
        b_max = kwargs_sys['b_max']
        
        def gen_y0():
            np.random.seed(42)
            ini_pop = np.random.rand()*p_max
            return {'population':ini_pop}
        
        def gen_params():
            np.random.seed(42)
            a = np.random.rand()*a_max
            b = np.random.rand()*b_max
            return {'a':a, 'b':b}, {}, {}
            
        @jit
        def eom(y, t, params, iparams, exparams):
            pop = y['population']
            a, b = params['a'], params['b']
            return {'population':a*pop+b}

        @jit
        def loss(ys, params, iparams, 
                        exparams, targets):
            pop = ys['population']
            t_pop = targets['population']
            return jnp.mean((pop-t_pop)**2)

        return eom, loss, gen_params, gen_y0, {}

    from adoptODE import simple_simulation, train_adoptODE
    kwargs_sys = {
        'p_max': 2,
        'a_max': 1,
        'b_max': 3,
        'N_sys': 1
    }
    kwargs_adoptODE = {'lr':3e-2, 'epochs':200}
    t_evals = np.linspace(0,5,10)
    np.random.seed(42)
    dataset = simple_simulation(
        define_system,
        t_evals,
        kwargs_sys,
        kwargs_adoptODE
    )

    from adoptODE import dataset_adoptODE
    data = np.array(
        [ 0.86, 1.66, 2.56, 3.59, 4.75, 6.08, 7.58, 9.28, 11.21, 13.40]
    ) # Observation of population, shape (10,)
    targets = {'population':data.reshape((1,10))}
    dataset2 = dataset_adoptODE(
        define_system,
        targets,
        t_evals,
        kwargs_sys,
        kwargs_adoptODE
    )
    _ = train_adoptODE(dataset2, print_interval=None)

Training can now be performed as before, with the difference that no error of the parameters can be given as the original parameters are unknown:

.. code-block:: python

    _ = train_adoptODE(dataset2)
    print('Found params: ', dataset2.params_train)

Will output

.. code-block:: python
    Found params:  {'a': Array(0.37454012, dtype=float32), 'b': Array(2.8521428, dtype=float32)}