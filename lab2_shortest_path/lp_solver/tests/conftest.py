import numpy as np
import pytest

from lp_solver.types import LPProblem, LPSolution

# ------------------ 可行且有最优解 ------------------
# A = [1 2 2 1 0 0;
#      2 1 2 0 1 0;
#      2 2 1 0 0 1];
# b = [20; 20; 20];
# c = [-10; -12; -12; 0; 0; 0];
# Solution x = [4;4;4;0;0;0], obj(min) = -136


@pytest.fixture
def lp_feasible_problem():
    A = np.array([[1., 2., 2., 1., 0., 0.],
                  [2., 1., 2., 0., 1., 0.],
                  [2., 2., 1., 0., 0., 1.]])
    b = np.array([20., 20., 20.])
    c = np.array([-10., -12., -12., 0., 0., 0.])  # min 形式
    bounds = (0.0, None)
    return LPProblem(
        c=c,
        A_eq=A,
        b_eq=b,
        A_ub=None,
        b_ub=None,
        bounds=bounds)

@pytest.fixture
def lp_feasible_solution():
    return LPSolution(
        success=True,
        status="optimal",
        nit=0,
        fun=-136.0,
        x=np.array([4., 4., 4., 0., 0., 0.]),
    )

# ------------------ 有冗余约束，仍有最优解 ------------------
# A = [1 2 3;
#      2 4 6;
#      1 1 1];
# b = [6; 12; 3];
# c = [1; 2; 3];
# Solution x = [0;3;0], obj(min)=6


@pytest.fixture
def lp_redundant_problem():
    A = np.array([[1., 2., 3.],
                  [2., 4., 6.],
                  [1., 1., 1.]])
    b = np.array([6., 12., 3.])
    c = np.array([1., 2., 3.])  # min 形式
    bounds = (0.0, None)
    lp = LPProblem(c=c, A_eq=A, b_eq=b, A_ub=None, b_ub=None, bounds=bounds)
    return lp

@pytest.fixture
def lp_redundant_solution():
    return LPSolution(
        success=True,
        status="optimal",
        nit=0,
        fun=6.0,
        x=np.array([0., 3., 0.]),
    )

# ------------------ 不可行 ------------------
# A = [1 0 0;
#      0 1 0;
#      0 0 1;
#      1 0 1];
# b = [2; 2; 2; 2];
# c = [1; 1; 1];
# 解释：前三行强制 x=[2,2,2]，但第四行要求 x1+x3=2 → 矛盾


@pytest.fixture
def lp_infeasible_problem():
    A = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.],
                  [1., 0., 1.]])
    b = np.array([2., 2., 2., 2.])
    c = np.array([1., 1., 1.])  # min 形式（无所谓）
    bounds = (0.0, None)
    lp = LPProblem(c=c, A_eq=A, b_eq=b, A_ub=None, b_ub=None, bounds=bounds)
    return lp

@pytest.fixture
def lp_infeasible_solution():
    return LPSolution(
        success=False,
        status="infeasible",
        nit=0,
        fun=np.nan,
        x=np.array([]),
    )

# ------------------ 无界 ------------------
# A = [ 1 -1;
#      -1  1];
# b = [0; 0];
# c = [-1; 0];   （min 形式；目标 = -x1）
# 约束等价于 x1 = x2，x>=0 → 目标可趋 -∞ → 无界


@pytest.fixture
def lp_unbounded_problem():
    A = np.array([[1., -1.],
                  [-1.,  1.]])
    b = np.array([0., 0.])
    c = np.array([-1., 0.])   # min 形式
    bounds = (0.0, None)
    lp = LPProblem(c=c, A_eq=A, b_eq=b, A_ub=None, b_ub=None, bounds=bounds)
    return lp

@pytest.fixture
def lp_unbounded_solution():
    return LPSolution(
        success=False,
        status="unbounded",
        nit=0,
        fun=np.nan,
        x=np.array([]),
    )
