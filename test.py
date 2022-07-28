import cupy as cp
import cubewalkers as cw
from timeit import default_timer as timer

cp.set_printoptions(edgeitems=5)

if __name__ == '__main__':
    print("INITIALIZING")
    rules = """#
A* = A
C* = 0
D* = 1
#E*=1 with prob 0.3, E*=0 with prob 0.7
E* = 1 & (0<<=0.3) | 0 & (0.3<<=1)
#"""

    experiment_string = """#
A,3,5,1
A,9,inf,!A
#"""

    myexperiment = cw.Experiment(experiment_string)
    mymodel = cw.Model(rules, experiment=myexperiment)
    # print(mymodel.code)
    N = mymodel.n_variables
    for T,W in [(15,3)]:
        initial_states = cp.array([[x%2 for x in range(N)] for y in range(W)], dtype=cp.bool_).T
        print(f'{initial_states=}')
        
        start = timer()
        mymodel.simulate_ensemble(n_time_steps=T, n_walkers=W,
                                  initial_states=initial_states,
                                  averages_only=False,
                                  maskfunction=cw.update_schemes.synchronous_PBN,
                                  threads_per_block=(32, 32))
        end = timer()
        print(f'{T=}, {W=}, {N=}, {end-start}s')

        for w in range(W):
            print(f'----- walker {w}')
            print(mymodel.trajectories[:,:,w].T)
            print('----')
