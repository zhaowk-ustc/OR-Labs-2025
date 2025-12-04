import numpy as np
from lp_solver.solver import solve


def test_feasible(lp_feasible_problem, lp_feasible_solution):
    lp = lp_feasible_problem
    expected = lp_feasible_solution
    res = solve(lp)
    assert res.status == expected.status
    assert res.success == expected.success
    # 比较目标值（允许小数差）
    assert np.isclose(res.fun, expected.fun, atol=1e-6)
    # 比较解向量长度和部分值
    assert res.x.shape == expected.x.shape
    assert np.allclose(res.x, expected.x, atol=1e-6)
