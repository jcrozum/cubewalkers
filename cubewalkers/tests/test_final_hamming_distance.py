import cubewalkers as cw

rules_zero = """
    A*=A|!B
    B*=A"""
    
rules_one ="""
    A*=!B
    B*=A"""

def test_source_final_hamming_distance_zero():
    mymodel = cw.Model(rules_zero,
                       n_time_steps=1000,
                       n_walkers=1000,
                       model_name='source_final_hamming_distance_zero')
    c = mymodel.source_final_hamming_distance('A',T_sample=100)
    assert c == 0.0
    
def test_final_hamming_distance_zero():
    mymodel = cw.Model(rules_zero,
                       n_time_steps=1000,
                       n_walkers=1000,
                       model_name='final_hamming_distance_zero')
    c = mymodel.final_hamming_distance(T_sample=100)
    assert c == 0.0

def test_source_final_hamming_distance_one():
    mymodel = cw.Model(rules_one,
                       n_time_steps=1000,
                       n_walkers=1000,
                       model_name='source_final_hamming_distance_one')
    c = mymodel.source_final_hamming_distance('A',T_sample=100)
    assert c == 1.0
    
def test_final_hamming_distance_one():
    mymodel = cw.Model(rules_one,
                       n_time_steps=1000,
                       n_walkers=1000,
                       model_name='final_hamming_distance_one')
    c = mymodel.final_hamming_distance(T_sample=100)
    assert c == 1.0
