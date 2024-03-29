import cupy as cp
import cubewalkers as cw

# It is important that these are all source nodes for the purposes of the tests below
test_rules = """
                A* = A
                B* = B
                C* = C
                D* = D
                E* = E
                """

initial_biases = """
                    A,0
                    B,1
                    C,0
                    D,1
                    E,0
                    """


def test_initial_condition_import():
    n_walkers = 1
    n_time_steps = 100
    test_model = cw.Model(
        test_rules,
        initial_biases=initial_biases,
        experiment=None,
        comment_char="#",
        n_time_steps=n_time_steps,
        n_walkers=n_walkers,
        model_name="test_simulation_asyncronous_model",
    )

    expected_initial_states = cp.array(
        [[x % 2 for x in range(test_model.n_variables)] for y in range(n_walkers)],
        dtype=cp.bool_,
    ).T

    assert (test_model.initial_states == expected_initial_states).all()


def test_initial_conditions_async():
    n_walkers = 100
    n_time_steps = 100
    test_model = cw.Model(
        test_rules,
        initial_biases=initial_biases,
        experiment=None,
        comment_char="#",
        n_time_steps=n_time_steps,
        n_walkers=n_walkers,
        model_name="test_simulation_asyncronous_model",
    )

    expected_initial_states = cp.array(
        [[x % 2 for x in range(test_model.n_variables)] for y in range(n_walkers)],
        dtype=cp.bool_,
    ).T

    test_model.simulate_ensemble(maskfunction=cw.update_schemes.asynchronous)

    assert all(
        [
            (test_model.trajectories[t, :, :] == expected_initial_states).all()
            for t in range(n_time_steps)
        ]
    )


def test_initial_conditions_sync():
    n_walkers = 100
    n_time_steps = 100
    test_model = cw.Model(
        test_rules,
        experiment=None,
        comment_char="#",
        n_time_steps=n_time_steps,
        n_walkers=n_walkers,
        model_name="test_simulation_syncronous_model",
    )

    initial_states = cp.array(
        [[x % 2 for x in range(test_model.n_variables)] for y in range(n_walkers)],
        dtype=cp.bool_,
    ).T
    test_model.initial_states = initial_states
    test_model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)

    assert all(
        [
            (test_model.trajectories[t, :, :] == initial_states).all()
            for t in range(n_time_steps)
        ]
    )
