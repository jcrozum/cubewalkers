import cupy as cp
import cubewalkers as cw

n = 10
w = 15

def test_update_mask():
    '''
    Tests few update masks.
    '''

    asynch = cw.update_schemes.asynchronous(None, n, w, None)
    updates_per_walker = cp.sum(asynch,axis=0)
    for update in updates_per_walker:
        assert update == 1, 'One node should be updated for each walker'

    synch = cw.update_schemes.synchronous(None, n, w, None)
    min = cp.min(synch,axis=0)
    for i in range(w):
        assert  min[i] == 1, 'All nodes must be updated for each walker'

    synch_pbn = cw.update_schemes.synchronous_PBN(None, n, w, None)
    max = cp.max(synch_pbn,axis=0)
    min = cp.min(synch_pbn,axis=0)
    for i in range(w):
        assert  max[i] != min[i], 'All nodes must use the different value for each walker'
        assert  min[i] != 0, 'All nodes must be updated for each walker'

    synch_pbn_dep = cw.update_schemes.synchronous_PBN_dependent(None, n, w, None)
    max = cp.max(synch_pbn_dep,axis=0)
    min = cp.min(synch_pbn_dep,axis=0)
    for i in range(w):
        assert  max[i] == min[i], 'All nodes must use the same value for each walker'
