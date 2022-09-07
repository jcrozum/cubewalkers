import cupy as cp
import cubewalkers as cw

test_rules = """
            A* = 0
            B* = !A
            C* = A&B
            """

true_derrida_coef = 2/3 # calculated by hand

def test_derrida_estimate_cf():
    test_derrida_model = cw.Model(test_rules,model_name='Derrida_test')
    estimated_derrida_coef = test_derrida_model.derrida_coefficient(n_walkers=10000)
    
    assert (abs(estimated_derrida_coef-true_derrida_coef)/true_derrida_coef < 0.1)