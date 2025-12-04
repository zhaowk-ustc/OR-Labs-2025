from lp_solver.solver import solve


def test_unbounded(lp_unbounded_problem, lp_unbounded_solution):
    lp = lp_unbounded_problem
    expected = lp_unbounded_solution
    res = solve(lp)
    assert res.status == expected.status
    assert res.success == expected.success
