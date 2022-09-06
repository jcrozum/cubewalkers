import cupy as cp
import cubewalkers as cw
from cupyx.profiler import benchmark

from cubewalkers.lookup_tables import *
cp.set_printoptions(edgeitems=5)


if __name__ == '__main__':

    from cana.datasets.bio import THALIANA
    from cubewalkers.lookup_tables import *
    ground_truth =[
        '101100110111011', 
        '110110110011011', 
        '100110110011011', 
        '111100110111011', 
        '010001000011100', 
        '010001001011100', 
        '001100110111011', 
        '000110110011001', 
        '000001000011100', 
        '000001001011100', 
    ]

    net = THALIANA()

    luts, inputs = cana2cupyLUT(net)

    testkern = lut_kernel(inputs, 'testkern')

    T = 1000
    W = 10000
    L = len(luts[0])
    N = net.Nnodes
    
    trajectories = simulate_lut_kernel(testkern, luts, N, T, W, L)
    for i,att in enumerate(reversed(sorted(set([tuple([int(y) for y in x]) for x in trajectories[-1].T])))):
        att_str = ''.join([str(x) for x in att])
        print(i,att_str,att_str in ground_truth)
    if i == 9:
        print("All attractors found!")
    else:
        print(f"Should have found 9, not {i}!")