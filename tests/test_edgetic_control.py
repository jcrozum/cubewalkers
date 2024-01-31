import cupy as cp
import cubewalkers as cw

rules = """
A*=0
B*=A
C*=A&B
"""

experiment_node_string = """
A,0,inf,1
"""

experiment_edge_string = """
A-->B,0,inf,1
"""

experiment_edge_string_alt = """
A-->B,0,inf,0
"""


def test_node_control():
    experiment_node = cw.Experiment(experiment_node_string)
    node_control_model = cw.Model(
        rules, experiment=experiment_node, model_name="node_control"
    )

    node_control_model.n_time_steps = 5
    node_control_model.n_walkers = 1

    node_control_model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)
    assert cp.all(
        node_control_model.trajectories[-1] == cp.array([[True], [True], [True]])
    )


def test_edge_control():
    experiment_edge = cw.Experiment(experiment_edge_string)
    edge_control_model = cw.Model(
        rules, experiment=experiment_edge, model_name="edge_control"
    )
    experiment_edge_alt = cw.Experiment(experiment_edge_string_alt)
    edge_control_model_alt = cw.Model(
        rules, experiment=experiment_edge_alt, model_name="edge_control_alt"
    )

    edge_control_model.n_time_steps = 5
    edge_control_model.n_walkers = 100
    edge_control_model_alt.n_time_steps = 5
    edge_control_model_alt.n_walkers = 100

    edge_control_model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)
    edge_control_model_alt.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)

    print(edge_control_model_alt.trajectories)

    assert cp.all(
        edge_control_model.trajectories[-1] == cp.array([[False], [True], [False]])
    )
    assert cp.all(
        edge_control_model_alt.trajectories[-1] == cp.array([[False], [False], [False]])
    )
