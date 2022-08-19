import cupy as cp
import cubewalkers as cw
from cupyx.profiler import benchmark

cp.set_printoptions(edgeitems=5)

if __name__ == '__main__':
    print("INITIALIZING")
    rules = """#
A* = !A
B* = A
C* = B
D* = D
#"""

    mymodel = cw.Model(rules)
    # print(mymodel.code)
    N = mymodel.n_variables
    for T, W in [(3, 1500)]:
        #initial_states = cp.array([[x%2 for x in range(N)] for y in range(W)], dtype=cp.bool_).T
        mymodel.n_time_steps = T
        mymodel.n_walkers = W
        
        print(
            benchmark(
                lambda : mymodel.simulate_ensemble(
                    maskfunction=cw.update_schemes.synchronous), 
                n_repeat=200)
            )  
        dynamical_impact = mymodel.dynamical_impact(['A','D'])
        
        print(f'{dynamical_impact=}')