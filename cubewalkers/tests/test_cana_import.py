# Cubewalkers import
from cubewalkers import conversions

## Feed-Forwward motif with NOT+AND
logic = {0: {'name': '3A', 'in':[], 'out':[0]},
         1: {'name': 'B+', 'in':[0], 'out':[1, 0]},
         2: {'name': '(C)', 'in':[0, 1], 'out':[0, 0, 0, 1]}}

cw_rules = "number_3A* = 0\nB_plus_* = !(number_3A)\n_C_* = number_3A&B_plus_"

def test_cana_import():
    
    # CANA import
    import cana.boolean_network as cana_bn
    
    BN = cana_bn.BooleanNetwork.from_dict(logic)
    cana_rules = conversions.network_rules_from_cana(BN)
    
    assert(cana_rules == cw_rules)