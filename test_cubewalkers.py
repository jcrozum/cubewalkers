import pytest

import cupy as cp

import cubewalkers as cw

test_rules ="""#
ABA* = ABA
C* = 0
D* = 1
E* = 1
#"""

experiment_string="""#
ABA,3,5,1
ABA,9,inf,!ABA
#"""

def test_rule_import():
    test_model=cw.Model(test_rules,
                        experiment=None,
                        comment_char='#',
                        n_time_steps=1,
                        n_walkers=1,
                        model_name="test_rule_import_model")
    
    expected_code = """extern "C" __global__
void test_rule_import_model(const bool* A__reserved_input,
        const bool* A__reserved_mask,
        bool* A__reserved_output,
        int t__reserved, int N__reserved, int W__reserved) {
    int w__reserved = blockDim.x * blockIdx.x + threadIdx.x;
    int n__reserved = blockDim.y * blockIdx.y + threadIdx.y;
    int a__reserved = w__reserved + n__reserved*W__reserved;
    if(n__reserved < N__reserved && w__reserved < W__reserved){
        if(A__reserved_mask[a__reserved]==1){
            if (n__reserved==0){A__reserved_output[a__reserved]=A__reserved_input[0*W__reserved+w__reserved];}
            if (n__reserved==1){A__reserved_output[a__reserved]=0;}
            if (n__reserved==2){A__reserved_output[a__reserved]=1;}
            if (n__reserved==3){A__reserved_output[a__reserved]=1;}
} else{A__reserved_output[a__reserved]=A__reserved_input[a__reserved];}}}"""
    
    assert test_model.code == expected_code

def test_rule_experiment_import():
    test_experiment = cw.Experiment(experiment_string)
    test_model=cw.Model(test_rules,
                        experiment=test_experiment,
                        comment_char='#',
                        n_time_steps=1,
                        n_walkers=1,
                        model_name="test_rule_experiment_import_model")
    
    expected_code = """extern "C" __global__
void test_rule_experiment_import_model(const bool* A__reserved_input,
        const bool* A__reserved_mask,
        bool* A__reserved_output,
        int t__reserved, int N__reserved, int W__reserved) {
    int w__reserved = blockDim.x * blockIdx.x + threadIdx.x;
    int n__reserved = blockDim.y * blockIdx.y + threadIdx.y;
    int a__reserved = w__reserved + n__reserved*W__reserved;
    if(n__reserved < N__reserved && w__reserved < W__reserved){
        if(A__reserved_mask[a__reserved]==1 || (n__reserved==0 && ((t__reserved >= 3 && t__reserved <= 5) || (t__reserved >= 9)))){
            if (n__reserved==0){A__reserved_output[a__reserved]=(t__reserved < 9) && (t__reserved < 3 || t__reserved > 5) && ( A__reserved_input[0*W__reserved+w__reserved] ) || (t__reserved >= 3 && t__reserved <= 5) && (1) || (t__reserved >= 9) && !A__reserved_input[0*W__reserved+w__reserved];}
            if (n__reserved==1){A__reserved_output[a__reserved]=0;}
            if (n__reserved==2){A__reserved_output[a__reserved]=1;}
            if (n__reserved==3){A__reserved_output[a__reserved]=1;}
} else{A__reserved_output[a__reserved]=A__reserved_input[a__reserved];}}}"""
    
    assert test_model.code == expected_code
    
def test_simulation_syncronous():
    test_experiment = cw.Experiment(experiment_string)
    test_model=cw.Model(test_rules,
                        experiment=test_experiment,
                        comment_char='#',
                        n_time_steps=100,
                        n_walkers=100,
                        model_name="test_simulation_syncronous_model")
    
    test_model.simulate_random_ensemble(maskfunction='synchronous')
    
    tail=cp.mean(test_model.trajectories[:,test_model.vardict['ABA'],:],axis=1)[4:]
    expected_tail = cp.array([1 for x in range(6)] + [x%2 for x in range(91)])
    
    assert all(tail == expected_tail)