"""
模块3：单纯形迭代
"""

from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np

from .types import Tableau, LPSolution
from .utils import DEFAULT_TOLERANCE as _TOL
from .exceptions import (
    InvalidInputError,
    InfeasibleError,
    UnboundedError,
    IterationLimitExceeded,
    NumericIssueError,
)

PivotRule = Literal["dantzig", "bland"]

# ---------- 小工具 ----------

def _ensure_tableau(T: Tableau, dual_tol: float, pivot_tol: float):
    if dual_tol < 0 or pivot_tol <= 0 or not np.isfinite(dual_tol) or not np.isfinite(pivot_tol):
        raise InvalidInputError("dual_tol/pivot_tol 非法", context={"dual_tol": dual_tol, "pivot_tol": pivot_tol})
    A, b, c = T.A, T.b, T.c
    if A.ndim != 2 or b.ndim != 1 or c.ndim != 1:
        raise InvalidInputError("Tableau 维度非法", context={"A_shape": A.shape, "b_shape": b.shape, "c_shape": c.shape})
    m, n = A.shape
    if b.size != m or c.size != n:
        raise InvalidInputError("Tableau 形状不匹配", context={"A": A.shape, "b": b.shape, "c": c.shape})
    if len(T.basis) != m:
        raise InvalidInputError("basis 长度应等于约束数 m", context={"len(basis)": len(T.basis), "m": m})
    if any(j < 0 or j >= n for j in T.basis):
        raise InvalidInputError("basis 含越界索引", context={"basis": T.basis, "n": n})
    if np.isnan(A).any() or np.isnan(b).any() or np.isnan(c).any() or not np.isfinite(T.z):
        raise InvalidInputError("Tableau 含 NaN/Inf 或 z 非有限")

def _choose_entering_max(c: np.ndarray, rule: PivotRule, dual_tol: float) -> Optional[int]:
    """选择入基列（max 问题，找 c_j > 0 的列）。返回列索引或 None（最优）。"""
    positive = np.where(c > dual_tol)[0]
    if positive.size == 0:
        return None
    if rule == "bland":
        return int(positive.min())
    # dantzig：选 c_j 最大
    j = int(positive[np.argmax(c[positive])])
    return j

def _choose_leaving(A_col: np.ndarray, b: np.ndarray, pivot_tol: float, rule: PivotRule) -> Optional[int]:
    """
    最小比值检验：i = argmin { b_i / a_ij | a_ij > pivot_tol }。
    若无 a_ij > pivot_tol，则无界。
    Bland 消歧：并列取最小行索引。
    """
    mask = A_col > pivot_tol
    if not np.any(mask):
        return None
    ratios = np.full_like(b, np.inf, dtype=float)
    ratios[mask] = b[mask] / A_col[mask]
    theta = np.min(ratios[mask])
    ties = np.where(np.isclose(ratios, theta, rtol=0.0, atol=1e-15))[0]
    return int(ties.min()) if rule == "bland" else int(np.argmin(ratios))

def _pivot_inplace(tab: Tableau, i: int, j: int, tol_clip: float) -> None:
    """
    在 (i,j) 处做枢轴更新 tableau（原地修改）：
      - 规范化主行：A[i,:] /= a_ij, b[i] /= a_ij
      - 对其它行 r：A[r,:] -= A[r,j] * A[i,:], b[r] -= A[r,j] * b[i]
      - 目标行：c -= c[j] * A[i,:], z += c[j] * b[i]
      - basis/nonbasis 交换 j 与 basis[i]
    """
    A, b, c = tab.A, tab.b, tab.c
    a_ij = A[i, j]
    if not np.isfinite(a_ij) or abs(a_ij) < tol_clip:
        raise NumericIssueError("枢轴过小或非有限", context={"i": i, "j": j, "a_ij": float(a_ij)})

    # 规范化主行
    A[i, :] = A[i, :] / a_ij
    b[i]    = b[i] / a_ij

    # 消去其它约束行
    m = A.shape[0]
    for r in range(m):
        if r == i:
            continue
        fac = A[r, j]
        if fac != 0.0:
            A[r, :] -= fac * A[i, :]
            b[r]    -= fac * b[i]

    # 更新目标行与 z
    fac = c[j]
    if fac != 0.0:
        c -= fac * A[i, :]
        tab.z += fac * b[i]

    # 数值清理：极小量置 0
    A[np.abs(A) < tol_clip] = 0.0
    c[np.abs(c) < tol_clip] = 0.0
    b[np.abs(b) < tol_clip] = 0.0

    # 交换基/非基
    old = tab.basis[i]
    tab.basis[i] = j
    try:
        tab.nonbasis.remove(j)
    except ValueError:
        # 若 j 不在 nonbasis，说明基集/非基集不一致
        raise InvalidInputError("nonbasis 与 basis 不一致", context={"enter": j, "basis_i_old": old})
    tab.nonbasis.append(old)

# ---------- 主过程 ----------

def iterate_simplex(
    tab: Tableau,
    max_iter: int = 10000,
    rule: PivotRule = "bland",
    dual_tol: float = 1e-9,
    pivot_tol: float = 1e-12,
) -> LPSolution:
    """
    单纯形法（标准形最大化）。
    结束条件：
      - 最优：所有 c_j <= dual_tol  → 返回 LPSolution(success=True, status="optimal", ...)
      - 无界：入基列存在，但列中无 a_ij>pivot_tol → 抛 UnboundedError
      - 迭代上限：抛 IterationLimitExceeded
    其余数值/输入问题：抛 InvalidInputError / NumericIssueError / InfeasibleError
    """
    if rule not in ("bland", "dantzig"):
        raise InvalidInputError("未知的入基规则", context={"rule": rule})

    # 原地工作：复制一份，避免改动来路数据
    T = tab.copy()
    _ensure_tableau(T, dual_tol, pivot_tol)

    # 先验可行性（b >= 0）：若为负，说明初始表不可行（模块2正常应保证 b>=0）
    if np.min(T.b) < -1e-9:
        raise InfeasibleError("初始表不可行：存在 b_i < 0", context={"min_b": float(np.min(T.b))})

    it = 0
    while it < max_iter:
        T.iteration = it

        # 1) 选入基列
        j_enter = _choose_entering_max(T.c, rule, dual_tol)
        if j_enter is None:
            # 最优
            x = _current_x(T)
            return LPSolution(True, "optimal", float(T.z), x, it)

        # 2) 选离基行（最小比值检验）
        i_leave = _choose_leaving(T.A[:, j_enter], T.b, pivot_tol, rule)
        if i_leave is None:
            # 无界
            x = _current_x(T)
            # 抛异常，让上层统一处理；也把当前 x/z 作为 context 便于排查
            raise UnboundedError("线性规划无界（进入列在约束中无正元素）",
                                 context={"enter": j_enter, "z": float(T.z)})

        # 3) 枢轴
        _pivot_inplace(T, i_leave, j_enter, tol_clip=max(1e-14, pivot_tol*1e-2))

        it += 1

    # 迭代上限
    x = _current_x(T)
    raise IterationLimitExceeded("达到迭代上限", context={"max_iter": max_iter, "z": float(T.z), "nit": it})

def _current_x(T: Tableau) -> np.ndarray:
    """从当前表恢复变量取值：基变量等于 b，其余为 0。"""
    x = np.zeros(T.n, dtype=float)
    for i, j in enumerate(T.basis):
        x[j] = T.b[i]
    return x
