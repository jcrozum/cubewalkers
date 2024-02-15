========
Examples
========

Here is a basic usage example to get you started running `cubewalkers`.

.. code-block:: python
    :caption: Basic usage

    import cubewalkers as cw

    # Define the rules. Various formats are accepted . . .
    rules = '''
    X,  !X
    Y,  X & !Z
    Z,  Y'''

    # For example, these two versions of `rules` are equivalent.
    rules = '''
    X* = !X
    Y* = X and not Z
    Z* = Y'''

    # Alternatively, rules can be imported from a text file by converting to a string, e.g.,
    # with open('/path/to/rule/file.bnet') as f:
    #  rules = f.read()

    # Convert the `rules` string into a `cubewalkers` model
    model = cw.Model(rules)

    # Specify the default number of time steps and walkers (independent simulations)
    model.n_time_steps = 25
    model.n_walkers = 5

    # Simulate the model ensemble (synchronous update is the default)
    model.simulate_ensemble()

    # The simulation results are stored in `model.trajectories` as a
    # `n_time_steps` x `n_variables` x `n_walkers` array of node values at
    # each timestep for each node and walker
    print(model.trajectories)

    # Simulate again, but this time,
    # 1) only keep the last 10 time steps
    # 2) only track the average values of nodes in the ensemble
    # 3) use asynchronous update
    # Points 1) and 2) are useful for memory efficiency in large simulations.
    model.simulate_ensemble(
    T_window = 10,
    averages_only = True,
    maskfunction = cw.update_schemes.asynchronous)

    # The `model.trajectories` variable is overwritten. It now contains
    # a `T_window` x `n_variables` array of averaged node values at each timestep
    # for each node, averaged across `n_walkers` simulations.
    print(model.trajectories)

Additional examples can be found in the notebooks in the `Examples and Tutorials`_ folder in the `cubewalkers` GitHub repository.

`Basic Usage.ipynb`_ shows how to run simple simulations and plot the results.

`CANA example.ipynb`_ shows how to import and simulate networks from `CANA`_.

`InfluenzaTest.ipynb`_ gives a demonstration of quantifying stochastic variations in model trajectories.

Further example usage can be found in our `cubewalkers-analysis respository`_, in which we conduct an in-depth analysis of the `Cell Collective`_.

.. _Examples and Tutorials: https://github.com/jcrozum/cubewalkers/tree/main/Examples%20and%20Tutorials
.. _Basic Usage.ipynb: https://github.com/jcrozum/cubewalkers/blob/main/Examples%20and%20Tutorials/Basic%20Usage.ipynb
.. _CANA example.ipynb: https://github.com/jcrozum/cubewalkers/blob/main/Examples%20and%20Tutorials/CANA%20Example.ipynb
.. _CANA: https://github.com/CASCI-lab/CANA
.. _InfluenzaTest.ipynb: https://github.com/jcrozum/cubewalkers/blob/main/Examples%20and%20Tutorials/InfluenzaTest.ipynb
.. _cubewalkers-analysis respository: https://github.com/kyuhyongpark/cubewalkers-analysis
.. _Cell Collective: https://cellcollective.org