import numpy as np
from lp_solver.solver import solve


def test_redundant_eq(lp_redundant_problem, lp_redundant_solution):
    lp = lp_redundant_problem
    expected = lp_redundant_solution
    res = solve(lp)
    assert res.status == expected.status
    assert res.success == expected.success
    assert np.isclose(res.fun, expected.fun, atol=1e-6)
    assert res.x.shape == expected.x.shape
    assert np.allclose(res.x, expected.x, atol=1e-6)
