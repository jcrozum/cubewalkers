import cupy as cp
import cubewalkers as cw

from cubewalkers.update_schemes import synchronous, asynchronous

cp.set_printoptions(edgeitems=5)

if __name__ == '__main__':
    rules = """
    A*=B
    B*=C
    C*=A"""
    mymodel = cw.Model(
        rules=rules,
        n_time_steps=1000,
        n_walkers=1000
    )
    print("Sync")
    for test in range(10):
        print(mymodel.quasicoherence(T_sample=100,maskfunction=synchronous))
        print(mymodel.quasicoherence(T_sample=100,maskfunction=synchronous,
                                     fuzzy_coherence=True))
    
    print("Async") 
    for test in range(5):
        print(mymodel.quasicoherence(T_sample=100,maskfunction=asynchronous))
        print(mymodel.quasicoherence(T_sample=100,maskfunction=asynchronous,
                                     fuzzy_coherence=True))
        
