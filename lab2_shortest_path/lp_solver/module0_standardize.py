"""
模块0：将一般形式的线性规划转为标准形。
"""
from typing import List, Optional, Tuple
import numpy as np
from .types import LPProblem, LPStandardForm, StandardFormTransform, BoundsType, Array
from .utils import DEFAULT_TOLERANCE
from .exceptions import InvalidInputError

# ---- 输入检查 ----
def _ensure_1d(a: np.ndarray, name: str):
    if a.ndim != 1:
        raise InvalidInputError(f"{name} 必须是一维数组", context={"name": name, "shape": a.shape})

def _ensure_shape(mat: np.ndarray, ncols: int, name: str):
    if mat.ndim != 2:
        raise InvalidInputError(f"{name} 必须是二维矩阵", context={"name": name, "shape": mat.shape})
    if mat.shape[1] != ncols:
        raise InvalidInputError(f"{name} 的列数必须与 c 的长度一致",
                                context={"name": name, "shape": mat.shape, "ncols_expected": ncols})

def _ensure_len(vec: np.ndarray, n: int, name: str):
    if vec.size != n:
        raise InvalidInputError(f"{name} 的长度与矩阵行数不匹配",
                                context={"name": name, "len": vec.size, "expect": n})

def _ensure_finite(arr: np.ndarray, name: str):
    if np.isnan(arr).any():
        raise InvalidInputError(f"{name} 含 NaN", context={"name": name})
    # 允许 bounds 为 ±inf，但 A/c/b 不应含 inf
    if np.isinf(arr).any():
        raise InvalidInputError(f"{name} 含 Inf", context={"name": name})

def _normalize_bounds(bounds: BoundsType, n: int) -> Tuple[Array, Array]:
    """把 bounds 统一成长度为 n 的 (lb, ub) 数组；None→±inf。"""
    if bounds is None:
        lb = np.full(n, -np.inf)
        ub = np.full(n,  np.inf)
    elif isinstance(bounds, tuple):
        lb_scalar, ub_scalar = bounds
        lb = np.full(n, -np.inf if lb_scalar is None else float(lb_scalar))
        ub = np.full(n,  np.inf if ub_scalar is None else float(ub_scalar))
    else:
        if len(bounds) != n:
            raise InvalidInputError("bounds 长度必须为 n", context={"len(bounds)": len(bounds), "n": n})
        lbs, ubs = zip(*bounds)
        lb = np.array([-np.inf if v is None else float(v) for v in lbs], dtype=float)
        ub = np.array([ np.inf if v is None else float(v) for v in ubs], dtype=float)

    # 基础合法性检查：逐分量上界不得小于下界
    bad = np.where(np.isfinite(lb) & np.isfinite(ub) & (lb > ub))[0]
    if bad.size:
        raise InvalidInputError("存在 lb > ub 的变量界", context={"indices": bad[:10].tolist(), "count": int(bad.size)})
    return lb, ub

def to_standard_form(
    lp: LPProblem,
    tol: float = DEFAULT_TOLERANCE
) -> Tuple[LPStandardForm, StandardFormTransform]:
    """
    将最小化形式的 LP 转为“最大化”标准形：
      max c_std^T x_std  s.t. A_std x_std = b_std,  x_std >= 0
    返回：(LPStandardForm, StandardFormTransform)
    """
    # ---- 基本输入检查与拷贝 ----
    if lp.c is None:
        raise InvalidInputError("c 不能为空")
    c = np.asarray(lp.c, dtype=float).copy()
    _ensure_1d(c, "c")
    _ensure_finite(c, "c")
    n = c.size

    Aeq = None if lp.A_eq is None else np.asarray(lp.A_eq, dtype=float).copy()
    beq = None if lp.b_eq is None else np.asarray(lp.b_eq, dtype=float).copy()
    Aub = None if lp.A_ub is None else np.asarray(lp.A_ub, dtype=float).copy()
    bub = None if lp.b_ub is None else np.asarray(lp.b_ub, dtype=float).copy()

    if Aeq is not None:
        _ensure_shape(Aeq, n, "A_eq")
        if beq is None:
            raise InvalidInputError("提供了 A_eq 但缺少 b_eq")
        beq = beq.reshape(-1)
        _ensure_len(beq, Aeq.shape[0], "b_eq")
        _ensure_finite(Aeq, "A_eq")
        _ensure_finite(beq, "b_eq")

    if Aub is not None:
        _ensure_shape(Aub, n, "A_ub")
        if bub is None:
            raise InvalidInputError("提供了 A_ub 但缺少 b_ub")
        bub = bub.reshape(-1)
        _ensure_len(bub, Aub.shape[0], "b_ub")
        _ensure_finite(Aub, "A_ub")
        _ensure_finite(bub, "b_ub")

    lb, ub = _normalize_bounds(lp.bounds, n)

    # 若无约束，构造空矩阵；确保 beq/bub 为 np.ndarray（非 None）
    if Aeq is None:
        Aeq = np.zeros((0, n))
        beq = np.zeros(0)
    else:
        beq = beq.reshape(-1) if beq is not None else np.zeros(Aeq.shape[0])

    if Aub is None:
        Aub = np.zeros((0, n))
        bub = np.zeros(0)
    else:
        bub = bub.reshape(-1) if bub is not None else np.zeros(Aub.shape[0])

    # === 第 0 步：处理固定变量 lb==ub ===
    z0 = 0.0
    keep_cols = []           # 非固定的原列索引
    fixed_values = [None]*n  # 记录固定值
    for j in range(n):
        if np.isfinite(lb[j]) and np.isfinite(ub[j]) and abs(lb[j]-ub[j]) <= tol:
            v = lb[j]
            fixed_values[j] = v
            z0 += c[j]*v
            if Aeq.shape[0] > 0:
                beq = beq - Aeq[:, j]*v
            if Aub.shape[0] > 0:
                bub = bub - Aub[:, j]*v
        else:
            keep_cols.append(j)

    # 如果有固定列，裁剪 A、c、lb、ub
    if len(keep_cols) < n:
        Aeq = Aeq[:, keep_cols] if Aeq.size else Aeq
        Aub = Aub[:, keep_cols] if Aub.size else Aub
        c   = c[keep_cols]
        lb  = lb[keep_cols]
        ub  = ub[keep_cols]
        n   = len(keep_cols)

    # === 第 1 步：为剩余变量做平移/拆分 ===
    var_names: List[str] = []
    c_trans: List[float] = []
    Aeq_cols: List[Array] = []
    Aub_cols: List[Array] = []
    kind: List[str] = []
    shift_list: List[float] = []
    keep_index: List[Optional[int]] = []
    split_index: List[Optional[Tuple[int, int]]] = []

    cap_rows: List[Array] = []
    cap_rhs: List[float] = []

    for j in range(n):
        col_eq = Aeq[:, j] if Aeq.size else np.zeros(0)
        col_ub = Aub[:, j] if Aub.size else np.zeros(0)
        cj     = c[j]

        # 再次校验单变量的上下界关系
        if np.isfinite(lb[j]) and np.isfinite(ub[j]) and lb[j] > ub[j] + tol:
            raise InvalidInputError("变量上下界冲突", context={"col": j, "lb": float(lb[j]), "ub": float(ub[j])})

        if np.isfinite(lb[j]):
            # 平移：x = y + lb_j，y>=0
            y_name = f"x{keep_cols[j] if len(keep_cols)>0 else j}_shift"
            var_names.append(y_name)
            c_trans.append(cj)
            Aeq_cols.append(col_eq)
            Aub_cols.append(col_ub)
            kind.append("shift")
            shift_list.append(lb[j])
            keep_index.append(len(var_names)-1)
            split_index.append(None)
            # 上界：y <= ub - lb  (若 ub 有限)
            if np.isfinite(ub[j]):
                cap_row = np.zeros(len(var_names))
                cap_row[-1] = 1.0
                cap_rows.append(cap_row)
                cap_rhs.append(ub[j] - lb[j])

            # 目标常数项 z0 += c_j * lb_j
            z0 += cj * lb[j]

        else:
            # lb = -inf → 拆分
            ypos_name = f"x{keep_cols[j] if len(keep_cols)>0 else j}_pos"
            yneg_name = f"x{keep_cols[j] if len(keep_cols)>0 else j}_neg"
            var_names += [ypos_name, yneg_name]
            c_trans   += [cj, -cj]
            Aeq_cols  += [col_eq, -col_eq]
            Aub_cols  += [col_ub, -col_ub]
            kind.append("split")
            shift_list.append(0.0)
            keep_index.append(None)
            split_index.append((len(var_names)-2, len(var_names)-1))

            # 上界：y+ - y- <= ub（若 ub 有限）
            if np.isfinite(ub[j]):
                cap_row = np.zeros(len(var_names))
                cap_row[-2] = 1.0
                cap_row[-1] = -1.0
                cap_rows.append(cap_row)
                cap_rhs.append(ub[j])

    # 组装 A_eq, A_ub 的列
    N_y = len(var_names)
    Aeq_new = np.column_stack(Aeq_cols) if Aeq_cols else np.zeros((Aeq.shape[0], 0))
    Aub_new = np.column_stack(Aub_cols) if Aub_cols else np.zeros((Aub.shape[0], 0))

    # 把上界 cap 变为 ≤ 约束，追加到 Aub_new
    if cap_rows:
        capA = np.vstack(cap_rows)              # (m_cap, N_y)
        capb = np.array(cap_rhs, dtype=float)   # (m_cap,)
        Aub_new = np.vstack([Aub_new, capA]) if Aub_new.size else capA
        if bub is None or getattr(bub, 'size', 0) == 0:
            bub = capb
        else:
            bub = np.concatenate((bub, capb))

    # === 第 2 步：把所有 ≤ 约束统一为等式，添加松弛变量 S >= 0 ===
    m_eq = Aeq_new.shape[0]
    m_ub = Aub_new.shape[0]
    if m_ub > 0:
        S = np.eye(m_ub)
        A_top = np.hstack([Aeq_new, np.zeros((m_eq, m_ub))]) if m_eq>0 else np.zeros((0, N_y+m_ub))
        A_bot = np.hstack([Aub_new, S])
        A_std = np.vstack([A_top, A_bot]) if m_eq>0 else A_bot
        if m_eq > 0:
            b_std = np.concatenate([np.asarray(beq, dtype=float), np.asarray(bub, dtype=float)])
        else:
            b_std = np.asarray(bub, dtype=float)
        c_std_min = np.concatenate([np.array(c_trans, dtype=float), np.zeros(m_ub)])
        var_names += [f"s_{i}" for i in range(m_ub)]
        base_hint = list(range(N_y, N_y + m_ub))  # 松弛列作为基的提示
    else:
        A_std = Aeq_new
        b_std = beq
        c_std_min = np.array(c_trans, dtype=float)
        base_hint = []

    # 终检：A/b/c 内不应含 NaN/Inf
    _ensure_finite(A_std, "A_std")
    _ensure_finite(b_std, "b_std")
    _ensure_finite(c_std_min, "c_std_min")

    # === 第 3 步：min → max（标准形以最大化为准）===
    c_std = -c_std_min
    std = LPStandardForm(A=A_std, b=b_std, c=c_std,
                         var_names=var_names, base_indices_hint=base_hint)

    # === 映射信息（对应原始 n_total 变量，下标以原顺序计）===
    n_total = len(lp.c)
    full_kind = ["fixed"]*n_total
    full_shift = [0.0]*n_total
    full_keep_index: List[Optional[int]] = [None]*n_total
    full_split_index: List[Optional[Tuple[int, int]]] = [None]*n_total
    full_fixed_value: List[Optional[float]] = [None]*n_total

    k = 0
    for j in range(n_total):
        if fixed_values[j] is not None:
            v = fixed_values[j]
            full_fixed_value[j] = float(v) if v is not None else None
            full_kind[j] = "fixed"
        else:
            full_kind[j] = kind[k]
            full_shift[j] = shift_list[k]
            if kind[k] == "shift":
                full_keep_index[j] = keep_index[k]
            else:
                full_split_index[j] = split_index[k]
            k += 1

    xfm = StandardFormTransform(
        kind=full_kind,
        shift=full_shift,
        keep_index=full_keep_index,
        split_index=full_split_index,
        fixed_value=full_fixed_value,
        z0=z0,
        flipped_to_max=True,
    )
    return std, xfm