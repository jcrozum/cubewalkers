import cubewalkers as cw

rules_zero = """
    A*=B
    B*=A"""

rules_one = """
    A*=!B
    B*=A"""


def test_source_quasicoherence_zero():
    mymodel = cw.Model(
        rules_zero,
        n_time_steps=1000,
        n_walkers=1000,
        model_name="source_quasicoherence_zero",
    )
    c = mymodel.source_quasicoherence("A", T_sample=100)
    assert c == 0.0


def test_quasicoherence_zero():
    mymodel = cw.Model(
        rules_zero,
        n_time_steps=1000,
        n_walkers=1000,
        model_name="source_quasicoherence_zero",
    )
    c = mymodel.quasicoherence(T_sample=100)
    assert c == 0.0


def test_source_quasicoherence_one():
    mymodel = cw.Model(
        rules_one,
        n_time_steps=1000,
        n_walkers=1000,
        model_name="source_quasicoherence_one",
    )
    c = mymodel.source_quasicoherence("A", T_sample=100)
    assert c == 1.0


def test_quasicoherence_one():
    mymodel = cw.Model(
        rules_one,
        n_time_steps=1000,
        n_walkers=1000,
        model_name="source_quasicoherence_one",
    )
    c = mymodel.quasicoherence(T_sample=100)
    assert c == 1.0
