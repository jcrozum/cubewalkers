import cubewalkers as cw

rules_zero = """
    A*=B
    B*=A"""
    
def test_source_coherence_zero():
    mymodel = cw.Model(rules_zero,
                       n_time_steps=1000,
                       n_walkers=1000,
                       model_name='source_coherence_zero')
    c = mymodel.source_coherence('A',T_sample=100)
    assert c == 0.0
    
def test_coherence_zero():
    mymodel = cw.Model(rules_zero,
                       n_time_steps=1000,
                       n_walkers=1000,
                       model_name='source_coherence_zero')
    c = mymodel.coherence(T_sample=100)
    assert c == 0.0