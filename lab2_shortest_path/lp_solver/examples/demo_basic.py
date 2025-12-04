"""
演示：一个简单的 LP 示例
- 在项目根目录运行本示例：
    python -m lp_solver.examples.demo_basic
"""

import time
import numpy as np
from lp_solver import solver, LPProblem

c = np.array([-3., -5.])
A_ub = np.array([[2., 1.],
                 [1., 2.]])
b_ub = np.array([6., 6.])
A_eq = None
b_eq = None
bounds = (0., None)

def demo():
    lp = LPProblem(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    print("Demo LP created:\n", lp, "\n")

    t0 = time.perf_counter()
    
    res = solver.solve(lp)

    time_total = time.perf_counter() - t0
    
    print("LP solved. Result:\n", res, "\n")

    print(f"[Total time] {time_total*1000:.3f} ms", "\n")

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