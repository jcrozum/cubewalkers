import cubewalkers as cw
import cupy as cp

def test_standard_heuristic():
    rules = """#
            A* = !A
            B* = A
            C* = B
            D* = D
            #"""

    mymodel = cw.Model(rules)
    mymodel.n_time_steps = 3
    mymodel.n_walkers = 30 # doesn't matter too much what we pick for n_walkers
    
    dynamical_impact = mymodel.dynamical_impact('A')
    
    expected_impact = cp.array([[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,0]])
    
    assert(cp.all(dynamical_impact == expected_impact))