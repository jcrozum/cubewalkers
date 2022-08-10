import cupy as cp
import cubewalkers as cw
from timeit import default_timer as timer

cp.set_printoptions(edgeitems=5)

if __name__ == '__main__':
    print("INITIALIZING")
    rules = """#
A* = !A
B* = A
C* = B
D* = D
E* = D | C
#"""

    mymodel = cw.Model(rules)
    # print(mymodel.code)
    N = mymodel.n_variables
    for T, W in [(5, 500)]:
        #initial_states = cp.array([[x%2 for x in range(N)] for y in range(W)], dtype=cp.bool_).T
        mymodel.n_time_steps = T
        mymodel.n_walkers = W
        

        start = timer()
        dynamical_impact = mymodel.dynamical_impact('A')
        end = timer()
        
        print(f'{dynamical_impact=}')