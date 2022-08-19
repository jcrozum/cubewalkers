import cubewalkers as cw
import cupy as cp

rules = """
        A* = !A
        B* = A
        C* = B
        D* = D
        """
        
def test_standard_heuristic():
    mymodel = cw.Model(rules)
    mymodel.n_time_steps = 3
    mymodel.n_walkers = 30  # doesn't matter too much what we pick for n_walkers

    dynamical_impact = mymodel.dynamical_impact('A')

    expected_impact = cp.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]])

    assert(cp.all(dynamical_impact == expected_impact))

def test_perturb_2():
    mymodel = cw.Model(rules)
    mymodel.n_time_steps = 3
    mymodel.n_walkers = 30  # doesn't matter too much what we pick for n_walkers

    dynamical_impact = mymodel.dynamical_impact(['A','D'])

    expected_impact = cp.array(
        [[1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    assert(cp.all(dynamical_impact == expected_impact))

def test_trajectory_variance():
    mymodel = cw.Model(rules)
    mymodel.n_time_steps = 3
    mymodel.n_walkers = 30

    trajectory_variance = mymodel.trajectory_variance([1, 1, 1, 1])
    assert all([(trajectory_variance[i] == 0.).sum() >= 4-i for i in range(4)])
