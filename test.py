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
        c = 0
        for name in mymodel.varnames:
            c += mymodel.source_quasicoherence(name,
                                          T_sample=100,
                                          maskfunction=synchronous)

        c /= mymodel.n_variables
        print(c)
    print("Async") 
    for test in range(5):
        print(mymodel.quasicoherence(T_sample=100,maskfunction=asynchronous))
        c = 0
        for name in mymodel.varnames:
            c += mymodel.source_quasicoherence(name,
                                          T_sample=100,
                                          maskfunction=asynchronous)

        c /= mymodel.n_variables
        print(c)
