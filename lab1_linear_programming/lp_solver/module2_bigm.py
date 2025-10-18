"""模块2：使用大M法构造初始基
"""

from typing import List, Tuple
import numpy as np

from .types import LPStandardForm, Tableau
from .utils import DEFAULT_TOLERANCE
from .exceptions import (
    InvalidInputError,
    NumericIssueError,
    PresolveError,
)

# ----------------- 内部小工具 -----------------

def _is_unit_col(col: np.ndarray, i: int, tol: float) -> bool:
    """判断一列是否是“在第 i 行为 1、其余行为 0”的单位列（带容差）。"""
    if abs(col[i] - 1.0) > tol:
        return False
    tmp = col.copy()
    tmp[i] = 0.0
    return np.max(np.abs(tmp)) <= tol

def _auto_big_m(c: np.ndarray) -> float:
    """
    自适应选择一个大 M：基于目标系数的数量级。
    经验值：1e6 * (1 + ||c||_inf)，既足够大，又不容易造成溢出。
    """
    scale = np.linalg.norm(c, ord=np.inf)
    scale_val = float(scale) if np.isfinite(scale) else 1.0
    return float(1e6 * (1.0 + scale_val))

# ----------------- 大 M 初始化 -----------------

def init_with_big_m(
    std: LPStandardForm,
    tol: float = DEFAULT_TOLERANCE,
    M: float | None = None,
) -> Tableau:
    """
    用大 M 法构造初始可行基解的单纯形表。

    输入:
        std: 标准形 (max)：max c^T x, s.t. A x = b, x >= 0
        tol: 数值容差（>0）
        M  : 大 M；None 时自动设置

    输出:
        Tableau: [A|b]、约化成本行 c、z、basis / nonbasis

    约定:
        - 初始基只使用“松弛列/人工列”的单位列。
        - 若某行 b_i < 0，先将该行整体乘 -1。
    """
    # -------- 基础输入检查 --------
    if tol <= 0 or not np.isfinite(tol):
        raise InvalidInputError("tol 必须为正实数", context={"tol": tol})

    A = np.asarray(std.A, dtype=float).copy()
    b = np.asarray(std.b, dtype=float).reshape(-1).copy()
    c = np.asarray(std.c, dtype=float).reshape(-1).copy()

    if A.ndim != 2 or b.ndim != 1 or c.ndim != 1:
        raise InvalidInputError("A 必须2维、b与c必须1维",
                                context={"A_shape": A.shape, "b_shape": b.shape, "c_shape": c.shape})
    m, n = A.shape
    if m != b.size:
        raise InvalidInputError("A 的行数必须等于 b 的长度",
                                context={"A_shape": A.shape, "b_len": b.size})
    if n != c.size:
        raise InvalidInputError("A 的列数必须等于 c 的长度",
                                context={"A_shape": A.shape, "c_len": c.size})
    # A/b/c 不应包含 NaN/Inf
    if np.isnan(A).any() or np.isnan(b).any() or np.isnan(c).any():
        raise InvalidInputError("A/b/c 含 NaN")
    if np.isinf(A).any() or np.isinf(b).any() or np.isinf(c).any():
        raise InvalidInputError("A/b/c 含 Inf")

    # 空系统（m=0）直接返回空表
    if m == 0:
        nonbasis = list(range(n))
        tab = Tableau(A=A, b=b, c=c.copy(), z=0.0, basis=[], nonbasis=nonbasis, iteration=0)
        # 再做一次有限性校验
        if not (np.isfinite(tab.A).all() and np.isfinite(tab.b).all() and np.isfinite(tab.c).all()):
            raise NumericIssueError("空系统初始化后出现非有限数")
        return tab

    if M is None:
        M = _auto_big_m(c)
    else:
        if not np.isfinite(M) or M <= 0:
            raise InvalidInputError("参数 M 必须为正且有限", context={"M": M})

    # --- 第0步：确保 b >= 0（行翻转） ---
    for i in range(m):
        if b[i] < -tol:
            A[i, :] *= -1.0
            b[i]    *= -1.0
    if np.isnan(A).any() or np.isnan(b).any():
        raise NumericIssueError("行翻转后出现 NaN")
    if np.isinf(A).any() or np.isinf(b).any():
        raise NumericIssueError("行翻转后出现 Inf")

    # --- 第1步：优先使用现有单位列；不足的用人工列补齐 ---
    basis: List[int] = [-1] * m
    use_existing = np.zeros(m, dtype=bool)

    hint = list(std.base_indices_hint or [])
    used_cols = set()
    for i in range(m):
        chosen = -1
        # 1) 先看 hint
        for j in hint:
            if j in used_cols or j < 0 or j >= n:
                continue
            if _is_unit_col(A[:, j], i, tol):
                chosen = j
                break
        # 2) 扫描所有列
        if chosen == -1:
            for j in range(n):
                if j in used_cols:
                    continue
                if _is_unit_col(A[:, j], i, tol):
                    chosen = j
                    break
        if chosen != -1:
            basis[i] = chosen
            used_cols.add(chosen)
            use_existing[i] = True
        # 否则留待人工列处理

    # --- 第2步：添加人工列 ---
    art_indices: List[int] = []
    art_rows: List[int] = []

    if not np.all(use_existing):
        need_rows = [i for i in range(m) if not use_existing[i]]
        k = len(need_rows)

        try:
            A_ext = np.hstack([A, np.zeros((m, k), dtype=float)]) if n > 0 else np.zeros((m, k), float)
        except Exception as e:
            raise NumericIssueError("扩展 A 失败（添加人工列）", context={"m": m, "n": n, "k": k}) from e

        try:
            c_ext = np.concatenate([c, -M * np.ones(k, dtype=float)])
        except Exception as e:
            raise NumericIssueError("扩展 c 失败（添加人工列）", context={"n": n, "k": k, "M": M}) from e

        for t, irow in enumerate(need_rows):
            col_idx = n + t
            A_ext[irow, col_idx] = 1.0
            basis[irow] = col_idx
            art_indices.append(col_idx)
            art_rows.append(irow)

        A = A_ext
        c = c_ext
        n = A.shape[1]

    # 若仍存在 -1，说明没能为某行找到/构造单位列 → 预处理失败
    if any(jb == -1 for jb in basis):
        raise PresolveError("无法构造初始可行基（缺少单位列或人工列构造失败）",
                            context={"rows_uncovered": [i for i, jb in enumerate(basis) if jb == -1]})

    # --- 第3步：构造初始目标行（约化成本）与 z ---
    obj_row = c.copy()
    z = 0.0
    for i in range(m):
        j_b = basis[i]
        if j_b < 0 or j_b >= n:
            raise PresolveError("基索引越界", context={"i": i, "j_b": j_b, "n": n})
        c_B = c[j_b]
        if abs(c_B) > tol:
            obj_row -= c_B * A[i, :]
            z += c_B * b[i]

    # --- 第4步：构造 nonbasis 并返回表 ---
    nonbasis = [j for j in range(n) if j not in basis]

    # 基本数值体检
    if not (np.isfinite(A).all() and np.isfinite(b).all() and np.isfinite(obj_row).all() and np.isfinite(z)):
        raise NumericIssueError("初始化后的表含非有限数")

    tab = Tableau(
        A=A,
        b=b,
        c=obj_row,
        z=float(z),
        basis=basis,
        nonbasis=nonbasis,
        iteration=0
    )
    return tab
