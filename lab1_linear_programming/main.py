import numpy as np
from scipy.optimize import linprog

# 例：max 3x1 + 5x2
c = np.array([-3, -5])  # 取负做最小化
A_ub = np.array([[2, 1],
                 [1, 2]])
b_ub = np.array([6, 6])
A_eq, b_eq = None, None
bounds = [(0, None), (0, None)]     # x1, x2 >= 0

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
              bounds=bounds, method="highs")
assert res.status == 0, res.message
x_star = res.x
obj_max = -res.fun
print(x_star, obj_max)  # → [2, 2], 16

print("Optimal solution:", res)