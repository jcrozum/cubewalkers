import cupy as cp
import cubewalkers as cw
from cupyx.profiler import benchmark

N = 100
W = 1000
T = 1000

print(
    benchmark(
        cw.update_schemes.synchronous,
        kwargs={"t": None, "n": N, "w": W, "a": None},
        n_repeat=T,
    )
)
print(
    benchmark(
        cw.update_schemes.asynchronous,
        kwargs={"t": None, "n": N, "w": W, "a": None},
        n_repeat=T,
    )
)
print(
    benchmark(
        cw.update_schemes.asynchronous_set,
        kwargs={"t": None, "n": N, "w": W, "a": None},
        n_repeat=T,
    )
)
print(
    benchmark(
        cw.update_schemes.synchronous_PBN,
        kwargs={"t": None, "n": N, "w": W, "a": None},
        n_repeat=T,
    )
)
print(
    benchmark(
        cw.update_schemes.synchronous_PBN_dependent,
        kwargs={"t": None, "n": N, "w": W, "a": None},
        n_repeat=T,
    )
)
print(
    benchmark(
        cw.update_schemes.asynchronous_PBN,
        kwargs={"t": None, "n": N, "w": W, "a": None},
        n_repeat=T,
    )
)
print(
    benchmark(
        cw.update_schemes.asynchronous_set_PBN,
        kwargs={"t": None, "n": N, "w": W, "a": None},
        n_repeat=T,
    )
)
print(
    benchmark(
        cw.update_schemes.asynchronous_set_PBN_dependent,
        kwargs={"t": None, "n": N, "w": W, "a": None},
        n_repeat=T,
    )
)
