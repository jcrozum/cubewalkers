import cupy as cp
import cubewalkers as cw

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
        c = cp.array([mymodel.source_coherence(name,
                                               T_sample=100,
                                               maskfunction=cw.update_schemes.synchronous) 
                      for name in mymodel.varnames])
        print(cp.mean(c))
    print("Async")
    for test in range(5):
        c = cp.array([mymodel.source_coherence(name,
                                               T_sample=100,
                                               maskfunction=cw.update_schemes.asynchronous) 
                      for name in mymodel.varnames])
        print(cp.mean(c))
  