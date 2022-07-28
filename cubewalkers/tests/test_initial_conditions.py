import cupy as cp
import cubewalkers as cw

# It is important that these are all source nodes for the purposes of the tests below
test_rules ="""#
A* = A
B* = B
C* = C
D* = D
E* = E
#"""

def test_initial_conditions_async():
    n_walkers = 100
    n_time_steps = 100
    test_model=cw.Model(test_rules,
                        experiment=None,
                        comment_char='#',
                        n_time_steps=n_time_steps,
                        n_walkers=n_walkers,
                        model_name="test_simulation_asyncronous_model")
    
    initial_conditions = cp.array(
        [[x%2 for x in range(test_model.n_variables)] for y in range(n_walkers)]).T
    
    test_model.simulate_ensemble(
        maskfunction=cw.update_schemes.asynchronous,
        initial_states=initial_conditions)
    
    assert all([(
        test_model.trajectories[t,:,:]==initial_conditions
        ).all() for t in range(n_time_steps)])
    
def test_initial_conditions_sync():
    n_walkers = 100
    n_time_steps = 100
    test_model=cw.Model(test_rules,
                        experiment=None,
                        comment_char='#',
                        n_time_steps=n_time_steps,
                        n_walkers=n_walkers,
                        model_name="test_simulation_syncronous_model")
    
    initial_conditions = cp.array(
        [[x%2 for x in range(test_model.n_variables)] for y in range(n_walkers)]).T
    
    test_model.simulate_ensemble(
        maskfunction=cw.update_schemes.synchronous,
        initial_states=initial_conditions)
    
    assert all([(
        test_model.trajectories[t,:,:]==initial_conditions
        ).all() for t in range(n_time_steps)])