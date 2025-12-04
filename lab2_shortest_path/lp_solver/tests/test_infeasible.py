from lp_solver.solver import solve


def test_infeasible(lp_infeasible_problem, lp_infeasible_solution):
    lp = lp_infeasible_problem
    expected = lp_infeasible_solution
    res = solve(lp)
    # 仅检查状态和 success 字段
    assert res.status == expected.status
    assert res.success == expected.success
