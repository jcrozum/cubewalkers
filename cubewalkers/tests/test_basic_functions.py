import cupy as cp
import cubewalkers as cw

test_rules ="""#
ABA* = ABA
C* = 0
D* = 1
E* = 1
#"""

experiment_string="""#
ABA,3,5,1
ABA,9,inf,!ABA
#"""
    
def test_simulation_syncronous():
    test_experiment = cw.Experiment(experiment_string)
    test_model=cw.Model(test_rules,
                        experiment=test_experiment,
                        comment_char='#',
                        n_time_steps=100,
                        n_walkers=100,
                        model_name="test_simulation_syncronous_model")
    
    test_model.simulate_random_ensemble(maskfunction='synchronous')
    
    tail=cp.mean(test_model.trajectories[:,test_model.vardict['ABA'],:],axis=1)[4:]
    expected_tail = cp.array([1 for x in range(6)] + [x%2 for x in range(91)])
    
    assert all(tail == expected_tail)
    
def test_simulation_syncronous_constant():
    test_experiment = cw.Experiment(experiment_string)
    test_model=cw.Model(test_rules,
                        experiment=test_experiment,
                        comment_char='#',
                        n_time_steps=100,
                        n_walkers=100,
                        model_name="test_simulation_syncronous_model")
    
    test_model.simulate_random_ensemble(maskfunction='synchronous')
    
    tail=cp.mean(test_model.trajectories[:,test_model.vardict['D'],:],axis=1)[4:]
    expected_tail = cp.array([1 for x in range(6)] + [1 for x in range(91)])
    
    assert all(tail == expected_tail)