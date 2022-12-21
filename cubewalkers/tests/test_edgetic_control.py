import cupy as cp
import cubewalkers as cw

rules = '''
A*=0
B*=A
C*=A&B
'''

experiment_node_string = '''
A,0,inf,1
'''

experiment_edge_string = '''
A-->B,0,inf,1
'''

# In case of contradiction, higher override should win
experiment_edge_string_conflict = '''
A-->B,0,inf,1
A-->B,0,inf,0
'''

def test_node_control():
    experiment_node = cw.Experiment(experiment_node_string)
    node_control_model = cw.Model(rules,experiment=experiment_node,model_name='node_control')
    
    node_control_model.n_time_steps=5
    node_control_model.n_walkers=1
    
    node_control_model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)
    assert(cp.all(node_control_model.trajectories[-1]==cp.array([[True],[True],[True]])))
    
def test_edge_control():
    experiment_edge = cw.Experiment(experiment_edge_string)
    edge_control_model = cw.Model(rules,experiment=experiment_edge,model_name='edge_control')
    
    edge_control_model.n_time_steps=5
    edge_control_model.n_walkers=1
    
    edge_control_model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)
    assert(cp.all(edge_control_model.trajectories[-1]==cp.array([[False],[True],[False]])))
    
def test_edge_control_conflict():
    experiment_edge_conflict = cw.Experiment(experiment_edge_string_conflict)
    edge_control_model_conflict = cw.Model(rules,experiment=experiment_edge_conflict,model_name='edge_control_conflict')
    
    edge_control_model_conflict.n_time_steps=5
    edge_control_model_conflict.n_walkers=1
    
    edge_control_model_conflict.simulate_ensemble(maskfunction=cw.update_schemes.synchronous)
    assert(cp.all(edge_control_model_conflict.trajectories[-1]==cp.array([[False],[True],[False]])))

    

