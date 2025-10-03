import cubewalkers as cw
import cupy as cp
from cana.boolean_network import BooleanNetwork

def test_probabilisticLUT():

    logic = {
        1: {'name': '1', 'in': [0, 1, 2], 'out': ['0', '1', '0', '1', '1', '0', '1', '0']},
        2: {'name': '2', 'in': [1, 2, 0], 'out': ['0', '1', '0', '1', '0', '0', '1', '0']},
        0: {'name': '0', 'in': [2, 0, 1], 'out': ['0', '1', '0', '1', '1', '0', '1', '0']}
    }
    initial_states = cp.array([[ True, False,  True],[ True, False,  True],[ True, False,  True]])

    net = BooleanNetwork.from_dict(logic)
    prob = 0.0

    prob_outs, prob_ins  = cw.conversions.cana2cupy_probabilisticLUT(net, prob)
    outs, ins = cw.conversions.cana2cupyLUT(net)
    prob_test_model = cw.Model(
        lookup_tables=prob_outs,
        node_regulators=prob_ins,
        n_time_steps=100,
        n_walkers=4,
        probabilistic_lut=True,
    )
    test_model = cw.Model(
        lookup_tables=outs,
        node_regulators=ins,
        n_time_steps=100,
        n_walkers=4,
        probabilistic_lut=False,
    )
    # lut_test_regulators = [[0], [0, 1], [2]]
    # test_lut = cp.array(
    #     [
    #         [0, 0, 0, 0],
    #         [1, 0, 1, 1],
    #         [0, 1, 0, 0],
    #     ],
    #     dtype=cp.bool_,
    # )
    # prob_test_lut = cp.array(
    #     [
    #         [0.1, 0.1, 0.1, 0.1],
    #         [0.9, 0.1, 0.9, 0.9],
    #         [0.1, 0.9, 0.1, 0.1],
    #     ],
    #     dtype=cp.float32,
    # )

    # initial_states = cp.array([[1, 0], [1, 0], [1, 0]])
    # test_model = cw.Model(
    #     lookup_tables=test_lut,
    #     node_regulators=lut_test_regulators,
    #     n_time_steps=100,
    #     n_walkers=3,)
    # prob_test_model = cw.Model(
    #     lookup_tables=prob_test_lut,
    #     node_regulators=lut_test_regulators,
    #     n_time_steps=100,
    #     n_walkers=3,
    #     probabilistic_lut=True,
    #     )


    prob_test_model.initial_states= initial_states
    test_model.initial_states= initial_states

    prob_test_model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous_PBN, T_window=100, averages_only=True)
    test_model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous_PBN, T_window=100, averages_only=True)

    print(prob_test_model.lookup_tables)
    print(test_model.lookup_tables)
    print(cp.sum(prob_test_model.trajectories))
    print(cp.sum(test_model.trajectories)) 
    # not sure how to test this, because the output varies even in non probabilistic LUTs where the array is boolean. can you help me with this?