import cubewalkers as cw
from cubewalkers.update_schemes import synchronous, asynchronous

rules_half = """
    A*=A&B
    B*=A&B"""

rules_zero = """
    A*=B
    B*=A"""

rules_fuzzy = """
    A*=B
    B*=C
    C*=A"""


def test_source_quasicoherence_half_sync_cf():
    mymodel = cw.Model(
        rules_half,
        n_time_steps=1000,
        n_walkers=1000,
        model_name="source_quasicoherence_half_sync_cf",
    )
    c = mymodel.source_quasicoherence("A", T_sample=100, maskfunction=synchronous)
    assert 0.55 > c > 0.45  # true value is 0.5 in synchronous


def test_source_quasicoherence_half_async_cf():
    mymodel = cw.Model(
        rules_half,
        n_time_steps=1000,
        n_walkers=1000,
        model_name="source_quasicoherence_half_async_cf",
    )
    c = mymodel.source_quasicoherence("A", T_sample=100, maskfunction=asynchronous)
    assert 0.55 > c > 0.45  # true value is 0.5 in asynchronous


def test_not_fuzzy_quasicoherence_sync_cf():
    mymodel = cw.Model(
        rules_fuzzy,
        n_time_steps=1000,
        n_walkers=1000,
        model_name="quasicoherence_not_fuzzy_sync_cf",
    )
    c = mymodel.quasicoherence(T_sample=100, maskfunction=synchronous)
    assert 0.55 > c > 0.45  # true value is 0.5 in synchronous


def test_fuzzy_quasicoherence_sync_cf():
    mymodel = cw.Model(
        rules_fuzzy,
        n_time_steps=1000,
        n_walkers=1000,
        model_name="quasicoherence_fuzzy_sync_cf",
    )
    c = mymodel.quasicoherence(
        T_sample=100, maskfunction=synchronous, fuzzy_coherence=True
    )
    assert 0.7 > c > 0.63  # true value is 2/3 in synchronous
