# LP Solver

## 环境配置

推荐使用 conda

```powershell
conda env create -f environment.yml
conda activate or25_lab1
```

## 最小示例

下面展示如何构造一个 `LPProblem` 并用 `solver.solve` 求解：

```python
import numpy as np
from lp_solver import solver, LPProblem

# 定义问题（例：min -3x_1-5x_2 s.t. 2x_1 + x_2 <= 6, x_1 + 2x_2 <= 6, x >= 0）
# 如果要求解最大值问题，需要手动转换为最小值
c = np.array([-3.0, -5.0])
A_ub = np.array([[2.0, 1.0],
				 [1.0, 2.0]])
b_ub = np.array([6.0, 6.0])

lp = LPProblem(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(0.0, None))

res = solver.solve(lp)

print("success:", res.success)
print("status:", res.status)
print("objective (min):", res.fun)
print("x:", res.x)
print("iterations:", res.nit)
```

返回值 `res` 的类型为 `LPSolution`，包含字段：
- `success` (bool)：是否成功得到最优解
- `status` (str)：状态说明（可能值示例："optimal", "infeasible", "unbounded", "iteration_limit", "invalid_input", "presolve_error", "numeric_issue", "unknown_error"）
- `fun` (float)：目标函数值（对 min 问题为最小值）
- `x` (ndarray)：解向量
- `nit` (int)：迭代次数

当问题不可行或无界时，`success` 为 False，`status` 会给出对应原因。

## 运行内置示例

仓库包含演示脚本，可直接运行：

```powershell
python -m lp_solver.examples.demo_basic
python -m lp_solver.examples.demo_step_by_step
```

- `demo_basic` 会构造一个简单的 LP、调用本求解器，并与 SciPy 的 `linprog` 比较。

- `demo_step_by_step.py` 用于按步骤显示每步的结果，适合调试。

## 测试

仓库包含一组单元测试，位于 `lp_solver/tests`。测试提供几个典型问题与期望解（包括可行最优、不可行、有冗余约束、无界等情形）。

在仓库根目录运行：

```powershell
# 运行所有测试（简洁输出）
python -m pytest -q

# 运行所有测试（详细输出）
python -m pytest -v

# 运行单个测试
python -m pytest -q lp_solver/tests/test_feasible.py
```

## 使用建议与注意事项

- 在调用前请确保输入数组尺寸正确（例如 `c` 长度和 `A` 的列数一致）。
- 默认变量界为 `(0.0, None)`，表示 x >= 0；如果需要逐变量界，请传入列表形式的 `bounds`。
