"""
模块1：秩检查与冗余约束移除
"""

from typing import List, Tuple
import numpy as np
from scipy.linalg import qr

from .types import LPStandardForm
from .utils import DEFAULT_TOLERANCE
from .exceptions import (
    InvalidInputError,
    InfeasibleError,
    NumericIssueError,
)

def _ensure_inputs(A: np.ndarray, b: np.ndarray, tol: float):
    if tol <= 0 or not np.isfinite(tol):
        raise InvalidInputError("tol 必须为正实数", context={"tol": tol})
    if A.ndim != 2:
        raise InvalidInputError("A 必须是二维矩阵", context={"A_shape": A.shape})
    if b.ndim != 1:
        raise InvalidInputError("b 必须是一维向量", context={"b_shape": b.shape})
    if A.shape[0] != b.size:
        raise InvalidInputError("A 的行数必须等于 b 的长度",
                                context={"A_shape": A.shape, "b_len": b.size})
    if np.isnan(A).any() or np.isnan(b).any():
        raise InvalidInputError("A 或 b 含 NaN")
    # 允许极小量；这里不允许 Inf（标准化后不该出现）
    if np.isinf(A).any() or np.isinf(b).any():
        raise InvalidInputError("A 或 b 含 Inf")

def _rowspace_fit(row: np.ndarray, A_keep: np.ndarray, b_keep: np.ndarray, tol: float) -> Tuple[float, float]:
    """
    计算 row 是否在 A_keep 的行空间内，并给出与 b 的一致性误差。
    解 alpha 使 alpha^T A_keep ≈ row  ⇔ A_keep^T alpha ≈ row^T（最小二乘）
    返回: (行空间残差, 预测的 b_hat)
    """
    if A_keep.size == 0:
        row_resid = float(np.linalg.norm(row, ord=np.inf))
        return row_resid, 0.0

    try:
        alpha, *_ = np.linalg.lstsq(A_keep.T, row.T, rcond=None)
    except Exception as e:
        raise NumericIssueError("最小二乘求解失败", context={"A_keep_T": A_keep.T.shape, "row": row.shape}) from e

    row_hat = (A_keep.T @ alpha).T
    row_resid = float(np.linalg.norm(row - row_hat, ord=np.inf))
    b_hat = float(alpha @ b_keep)
    return row_resid, b_hat

def _rebuild_slack_hints(A: np.ndarray, tol: float) -> List[int]:
    """重新识别单位列作为 base_indices_hint。"""
    m, n = A.shape
    hints: List[int] = []
    for j in range(n):
        col = A[:, j]
        if m == 0:
            break
        i_star = int(np.argmax(np.abs(col)))
        if abs(col[i_star] - 1.0) <= tol:
            tmp = col.copy()
            tmp[i_star] = 0.0
            if np.max(np.abs(tmp)) <= tol:
                hints.append(j)
    return hints

# ---------- 对外函数（默认 RRQR） ----------

def prune_redundant_rows(std: LPStandardForm, tol: float = DEFAULT_TOLERANCE) -> LPStandardForm:
    return prune_redundant_rows_rrqr(std, tol)

# ---------- SVD 增量式版本 ----------

def _matrix_rank_svd(M: np.ndarray, tol: float) -> int:
    """用 SVD 估计矩阵秩，阈值随规模缩放。"""
    if M.size == 0:
        return 0
    try:
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    except Exception as e:
        raise NumericIssueError("SVD 失败", context={"shape": M.shape}) from e
    if s.size == 0:
        return 0
    thresh = tol * max(M.shape) * (s[0] if s[0] > 0 else 1.0)
    return int(np.sum(s > thresh))

def prune_redundant_rows_svd(std: LPStandardForm, tol: float = DEFAULT_TOLERANCE) -> LPStandardForm:
    A = np.asarray(std.A, dtype=float)
    b = np.asarray(std.b, dtype=float).reshape(-1)
    _ensure_inputs(A, b, tol)

    m, n = A.shape
    keep_rows: List[int] = []

    # 0) 全零行处理
    for i in range(m):
        if np.max(np.abs(A[i, :])) <= tol:
            if abs(b[i]) > tol:
                raise InfeasibleError("不可行：出现零行但 b 非零", context={"row": i, "b": float(b[i])})
        else:
            keep_rows.append(i)

    if not keep_rows:
        # A 全零且 b≈0：空系统
        A2 = np.zeros((0, n), dtype=float)
        b2 = np.zeros(0, dtype=float)
        hints = _rebuild_slack_hints(A2, tol)
        return LPStandardForm(A=A2, b=b2, c=std.c.copy(),
                              var_names=list(std.var_names),
                              base_indices_hint=hints)

    # 1) 增量式选行
    A_keep = A[keep_rows[:1], :]
    b_keep = b[keep_rows[:1]]
    rank_keep = _matrix_rank_svd(A_keep, tol)
    selected = [keep_rows[0]]

    for i in keep_rows[1:]:
        A_try = np.vstack([A_keep, A[i, :]])
        rank_try = _matrix_rank_svd(A_try, tol)
        if rank_try > rank_keep:
            A_keep = A_try
            b_keep = np.concatenate([b_keep, [b[i]]])
            rank_keep = rank_try
            selected.append(i)
        else:
            row_resid, b_hat = _rowspace_fit(A[i, :], A_keep, b_keep, tol)
            if row_resid > max(tol, 10*tol*np.linalg.norm(A[i,:], ord=np.inf)):
                # 数值不稳：保守纳入
                A_keep = A_try
                b_keep = np.concatenate([b_keep, [b[i]]])
                rank_keep = _matrix_rank_svd(A_keep, tol)
                selected.append(i)
            else:
                if abs(b[i] - b_hat) > (10 * tol * (1.0 + abs(b[i]))):
                    raise InfeasibleError(
                        "不可行：线性相关行对 RHS 不一致",
                        context={"row": i, "b": float(b[i]), "b_hat": float(b_hat), "row_resid": row_resid}
                    )
                # 真冗余：跳过

    # 2) 输出
    A2 = A[selected, :]
    b2 = b[selected]
    hints = _rebuild_slack_hints(A2, tol)
    return LPStandardForm(A=A2, b=b2, c=std.c.copy(),
                          var_names=list(std.var_names),
                          base_indices_hint=hints)

# ---------- RRQR 一次性选行版本 ----------

def prune_redundant_rows_rrqr(std: LPStandardForm, tol: float = DEFAULT_TOLERANCE) -> LPStandardForm:
    A = np.asarray(std.A, dtype=float)
    b = np.asarray(std.b, dtype=float).reshape(-1)
    _ensure_inputs(A, b, tol)

    m, n = A.shape
    if m == 0:
        return std  # 空系统

    # 0) 全零行处理
    nonzero_rows: List[int] = []
    for i in range(m):
        if np.max(np.abs(A[i, :])) <= tol:
            if abs(b[i]) > tol:
                raise InfeasibleError("不可行：出现零行但 b 非零", context={"row": i, "b": float(b[i])})
        else:
            nonzero_rows.append(i)

    if not nonzero_rows:
        A2 = np.zeros((0, n), dtype=float)
        b2 = np.zeros(0, dtype=float)
        hints = _rebuild_slack_hints(A2, tol)
        return LPStandardForm(A=A2, b=b2, c=std.c.copy(),
                              var_names=list(std.var_names),
                              base_indices_hint=hints)

    A0 = A[nonzero_rows, :]
    b0 = b[nonzero_rows]

    # 1) RRQR（对 A0^T）
    try:
        qr_res = qr(A0.T, mode="economic", pivoting=True)  # A0^T = Q R P^T
        # scipy.linalg.qr has overloads that some type checkers consider ambiguous;
        # handle possible return shapes robustly at runtime.
        if isinstance(qr_res, tuple) and len(qr_res) == 3:
            Q, R, piv = qr_res
        elif isinstance(qr_res, tuple) and len(qr_res) == 2:
            Q, R = qr_res
            # No pivot array returned despite requesting pivoting: assume natural order
            piv = np.arange(A0.T.shape[1], dtype=int)
        else:
            # Fallback: attempt to unpack and ignore static type checks if needed
            Q, R, piv = qr_res  # type: ignore
    except Exception as e:
        raise NumericIssueError("QR 分解失败（pivoting=True）", context={"A0_T": A0.T.shape}) from e

    diag = np.abs(np.diag(R))
    if diag.size == 0:
        r = 0
    else:
        thresh = tol * max(A0.shape) * (diag[0] if diag[0] > 0 else 1.0)
        r = int(np.sum(diag > thresh))

    keep_local = np.sort(piv[:r]) if r > 0 else np.array([], dtype=int)
    keep_rows = [nonzero_rows[i] for i in keep_local]

    # 2) 一致性检查（对相关行）
    dependent_rows = [i for i in nonzero_rows if i not in keep_rows]
    if len(keep_rows) == 0:
        # 极端病态：保守保留第一行，避免全删
        keep_rows = [nonzero_rows[0]]
        dependent_rows = [i for i in nonzero_rows if i != keep_rows[0]]

    A_keep = A[keep_rows, :]
    b_keep = b[keep_rows]

    for i in dependent_rows:
        row = A[i, :]
        row_resid, b_hat = _rowspace_fit(row, A_keep, b_keep, tol)
        if row_resid > max(tol, 10 * tol * np.linalg.norm(row, ord=np.inf)):
            # 数值不稳：保守纳入
            keep_rows.append(i)
            A_keep = A[keep_rows, :]
            b_keep = b[keep_rows]
            continue
        if abs(b[i] - b_hat) > (10 * tol * (1.0 + abs(b[i]))):
            raise InfeasibleError(
                "不可行：线性相关行对 RHS 不一致",
                context={"row": i, "b": float(b[i]), "b_hat": float(b_hat), "row_resid": row_resid}
            )
        # 一致：冗余，丢弃

    # 3) 输出
    keep_rows = sorted(set(keep_rows))
    A2 = A[keep_rows, :]
    b2 = b[keep_rows]
    hints = _rebuild_slack_hints(A2, tol)

    return LPStandardForm(A=A2, b=b2, c=std.c.copy(),
                          var_names=list(std.var_names),
                          base_indices_hint=hints)