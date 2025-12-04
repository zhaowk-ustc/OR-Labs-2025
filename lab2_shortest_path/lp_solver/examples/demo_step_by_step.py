"""
演示：LP求解器各模块的逐步执行过程。
- 在项目根目录运行本示例：
    python -m lp_solver.examples.demo_step_by_step
"""

import time
import numpy as np
from lp_solver.types import LPProblem, BoundsType
from lp_solver.module0_standardize import to_standard_form
from lp_solver.module1_redundancy import prune_redundant_rows
from lp_solver.module2_bigm import init_with_big_m
from lp_solver.module3_simplex import iterate_simplex
from lp_solver.module4_postprocess import recover_solution

c = np.array([-3., -5.])
A_ub = np.array([[2., 1.],
                 [1., 2.]])
b_ub = np.array([6., 6.])
A_eq = None
b_eq = None
bounds: BoundsType = [(0., None), (0., None)]

def demo():
    lp = LPProblem(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    print("Demo LP created:\n", lp, "\n")

    t0 = time.perf_counter()
    std, xfm = to_standard_form(lp)
    print("STEP 0: Standard form:\n", std, "\n")
    time_step0 = time.perf_counter() - t0
    print(f"[STEP 0 to_standard_form] {time_step0*1000:.3f} ms", "\n")

    t0 = time.perf_counter()
    std_pruned = prune_redundant_rows(std)
    print("STEP 1: After redundancy pruning:\n", std_pruned, "\n")
    time_step1 = time.perf_counter() - t0
    print(f"[STEP 1 prune_redundant_rows] {time_step1*1000:.3f} ms", "\n")

    t0 = time.perf_counter()
    tab = init_with_big_m(std_pruned)
    print("STEP 2: Initial tableau with Big M:\n", tab, "\n")
    time_step2 = time.perf_counter() - t0
    print(f"[STEP 2 init_with_big_m] {time_step2*1000:.3f} ms", "\n")

    t0 = time.perf_counter()
    sol = iterate_simplex(tab, rule="bland")
    print("STEP 3: Final result after Simplex iterations:\n", sol, "\n")
    time_step3 = time.perf_counter() - t0
    print(f"[STEP 3 iterate_simplex] {time_step3*1000:.3f} ms", "\n")

    t0 = time.perf_counter()
    sol_recovered = recover_solution(sol, xfm, sense="min")
    print("STEP 4: Recovered original solution:\n", sol_recovered, "\n")
    time_step4 = time.perf_counter() - t0
    print(f"[STEP 4 recover_solution] {time_step4*1000:.3f} ms", "\n")


    print(f"Recovered original min objective: {sol_recovered.fun:.6g}")
    print(f"Recovered original x: {sol_recovered.x}\n")

    time_total = time_step0 + time_step1 + time_step2 + time_step3 + time_step4
    print(f"[Total time for all steps] {time_total*1000:.3f} ms", "\n")

def demo_scipy():
    try:
        from scipy.optimize import linprog
    except Exception as e:
        print("[Skip SciPy] SciPy 未安装或不可用：", e)
        return

    t0 = time.perf_counter()
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    time_scipy = time.perf_counter() - t0
    print("---- SciPy linprog (HiGHS) ----")
    print("success:", res.success, " status:", res.status)
    print("message:", getattr(res, "message", ""))
    print("fun (min):", res.fun)
    print("x:", res.x)
    print(f"[SciPy linprog] {time_scipy*1000:.3f} ms", "\n")

if __name__ == '__main__':
    t0_total = time.perf_counter()
    demo()
    print("\n================ Compare with SciPy ================\n")
    demo_scipy()