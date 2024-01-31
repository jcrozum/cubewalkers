import cubewalkers as cw

rules = """
        A* = !A
        B* = A
        C* = B
        D* = D
        """


def test_trajectory_variance_cf():
    mymodel = cw.Model(rules)
    mymodel.n_time_steps = 3
    mymodel.n_walkers = 30

    trajectory_variance = mymodel.trajectory_variance([1, 1, 1, 1])
    assert (trajectory_variance >= 0).all()  # fail prob ~(1/N)**W
