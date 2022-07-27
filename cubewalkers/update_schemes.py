from __future__ import annotations
import cupy as cp


def general_asynchronous_update_mask(t: int, n: int, w: int, a: int) -> cp.array:
    uinds = cp.random.randint(0, w, (n,))
    z = cp.zeros((n, w), dtype=cp.float32)
    z[cp.arange(n), uinds] = cp.float32(1)
    return z


def fully_asynchronous_update_mask(t: int, n: int, w: int, a: int) -> cp.array:
    return cp.random.choice([cp.bool_(0), cp.bool_(1)], (n, w))


def synchronous_update_mask(t: int, n: int, w: int, a: int) -> cp.array:
    return cp.ones((n, w), dtype=cp.float32)

def synchronous_update_mask_PBN(t: int, n: int, w: int, a: int) -> cp.array:
    return cp.random.random((n, w), dtype=cp.float32)
