import cupy as cp
import cubewalkers as cw

from cubewalkers.update_schemes import synchronous, asynchronous

from cana.datasets.bio import THALIANA
from cubewalkers.conversions import cana2cupyLUT
cp.set_printoptions(edgeitems=5)

if __name__ == '__main__':
    net = THALIANA()
    luts, inputs = cana2cupyLUT(net)
    mymodel = cw.Model(
        lookup_tables=luts,
        node_regulators=inputs,
        n_time_steps=1000,
        n_walkers=1000
    )
    print("Sync")
    for test in range(5):
        print(mymodel.quasicoherence(T_sample=100,maskfunction=synchronous))
        print(mymodel.quasicoherence(T_sample=100,maskfunction=synchronous,
                                     fuzzy_coherence=True))
    print("Async") 
    for test in range(5):
        print(mymodel.quasicoherence(T_sample=100,maskfunction=asynchronous))
        print(mymodel.quasicoherence(T_sample=100,maskfunction=asynchronous,
                                     fuzzy_coherence=True))
        
