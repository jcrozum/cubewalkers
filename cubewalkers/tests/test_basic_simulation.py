import cupy as cp
import cubewalkers as cw

test_rules = """
            A* = A
            C* = 0
            D* = 1
            E* = 1
            """

experiment_string = """
                    A,3,5,1
                    A,9,inf,!A
                    """

unforced_experiment_test_model = """
                                A, 0
                                B, A
                                """

unforced_experiment_string = """
                             A*,1,inf,1
                             """

def test_simulation_syncronous():
    test_experiment = cw.Experiment(experiment_string)
    test_model = cw.Model(test_rules,
                          experiment=test_experiment,
                          comment_char='#',
                          n_time_steps=100,
                          n_walkers=100,
                          model_name="test_simulation_syncronous_model")

    test_model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)

    tail = cp.mean(
        test_model.trajectories[:, test_model.vardict['A'], :], axis=1)[4:]
    expected_tail = cp.array([1 for x in range(6)] +
                             [x % 2 for x in range(91)])

    assert all(tail == expected_tail)

def test_lut_simulation_synchronous():
    lut_test_rules="""
    A*=0
    B*=A or not B
    C*=C
    """
    lut_test_regulators =[
        [],
        [0,1],
        [2]
    ]
    lut_test_lut=cp.array([
        [0,0,0,0],
        [1,0,1,1],
        [0,1,0,0],
    ],dtype=cp.bool_)
    
    lut_test_initial_states=cp.array([
        [1,0],
        [1,0],
        [1,0]
    ])
    test_model_with_rules = cw.Model(lut_test_rules,
                          comment_char='#',
                          n_time_steps=100,
                          n_walkers=2,
                          model_name="lut_test_with_rules")
    test_model_with_lut = cw.Model(
        lookup_tables=lut_test_lut,
        node_regulators=lut_test_regulators,
        n_time_steps=100,
        n_walkers=2,
        model_name="lut_test_with_lut")
    
    test_model_with_rules.initial_states=lut_test_initial_states
    test_model_with_lut.initial_states=lut_test_initial_states
    
    test_model_with_rules.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)
    test_model_with_lut.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)
    
    assert(cp.sum(test_model_with_rules.trajectories!=test_model_with_lut.trajectories)==0)
 

def test_simulation_syncronous_constant():
    test_experiment = cw.Experiment(experiment_string)
    test_model = cw.Model(test_rules,
                          experiment=test_experiment,
                          comment_char='#',
                          n_time_steps=100,
                          n_walkers=100,
                          model_name="test_simulation_syncronous_model")

    test_model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)

    tail = cp.mean(
        test_model.trajectories[:, test_model.vardict['D'], :], axis=1)[4:]
    expected_tail = cp.array([1 for x in range(6)] + [1 for x in range(91)])

    assert all(tail == expected_tail)

def test_unforced_experiment():
    unforced_experiment = cw.Experiment(unforced_experiment_string)
    test_model = cw.Model(unforced_experiment_test_model,
                          experiment=unforced_experiment,
                          comment_char='#',
                          n_time_steps=10,
                          n_walkers=100,
                          model_name="test_unforced_experiment")

    def bespoke_updater(t,n,w,a):
        mask = cp.ones((n, w), dtype=cp.float32)
        if t >= 1:
            mask[0,:] = cp.float32(0)
        return mask

    test_model.simulate_ensemble(maskfunction=bespoke_updater)

    tail = cp.mean(
        test_model.trajectories[:, test_model.vardict['A'], :], axis=1)[4:]
    expected_tail = cp.array([0 for x in range(7)])

    assert all(tail == expected_tail)