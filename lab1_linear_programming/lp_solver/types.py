from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Literal
import numpy as np

# 简化类型声明
Array = np.ndarray
OptFloat = Optional[float]
OptArray = Optional[Array]
BoundsType = Optional[Union[Tuple[OptFloat, OptFloat], List[Tuple[OptFloat, OptFloat]]]]
StatusType = Literal[
    "optimal", 
    "unbounded", 
    "iteration_limit", 
    "infeasible",
    "invalid_input",
    "presolve_error",
    "unknown_error",
    "numeric_issue"
    ]


# ---------- 数组/界限的友好字符串 ----------

def _arr_str(a: OptArray, prefix: str = "", max_elems: int = 8, precision: int = 6) -> str:
    if a is None:
        return f"{prefix}None"
    a = np.asarray(a)
    body = np.array2string(
        a,
        precision=precision,
        suppress_small=True,
        threshold=max_elems,
        max_line_width=120,
    )
    return f"{prefix}\nshape={a.shape}, dtype={a.dtype}\n{body}"

def _bounds_str(bounds: BoundsType, n: int | None = None, preview: int = 3) -> str:
    if bounds is None:
        return "bounds: None"
    if isinstance(bounds, tuple):
        lb, ub = bounds
        return f"bounds: all vars -> (lb={lb}, ub={ub})"
    # list
    head = ", ".join(f"({lb},{ub})" for lb, ub in bounds[:preview])
    more = "" if len(bounds) <= preview else f", ... x{len(bounds)-preview} more"
    return f"bounds: [{head}{more}]"

def _head_indices(xs: List[int], k: int = 8) -> str:
    if not xs:
        return "[]"
    if len(xs) <= k:
        return "[" + ", ".join(map(str, xs)) + "]"
    return "[" + ", ".join(map(str, xs[:k])) + f", ... x{len(xs)-k} more]"

# ---------- 类型定义 ----------

@dataclass
class LPProblem:
    """原始 LP：min c^T x s.t. A_ub x <= b_ub, A_eq x = b_eq, bounds"""
    c: Array              # (n,)
    A_ub: OptArray = None  # (m_ub,n) 或 None
    b_ub: OptArray = None  # (m_ub,) 或 None
    A_eq: OptArray = None  # (m_eq,n) 或 None
    b_eq: OptArray = None  # (m_eq,) 或 None
    bounds: BoundsType = (0.0, None)   # 变量界

    def __str__(self) -> str:
        n = int(self.c.size)
        m_ub = 0 if self.A_ub is None else int(self.A_ub.shape[0])
        m_eq = 0 if self.A_eq is None else int(self.A_eq.shape[0])
        parts = [
            f"<LPProblem: min c^T x | A_ub * x <= b ({m_ub} rows), A_eq * x = b_eq ({m_eq} rows) | n = {n}>",
            _arr_str(self.c, "c: "),
            _arr_str(self.A_ub, "A_ub: "),
            _arr_str(self.b_ub, "b_ub: "),
            _arr_str(self.A_eq, "A_eq: "),
            _arr_str(self.b_eq, "b_eq: "),
            _bounds_str(self.bounds, n)
        ]
        return "\n".join(parts)


@dataclass
class LPStandardForm:
    """标准形：max c^T x, s.t. A x = b, x >= 0"""
    A: np.ndarray       # (m, N)
    b: np.ndarray       # (m,)
    c: np.ndarray       # (N,)
    var_names: List[str]
    base_indices_hint: List[int]    # 松弛变量索引的提示

    def __str__(self) -> str:
        m, N = self.A.shape
        vn = len(self.var_names)
        hint = _head_indices(self.base_indices_hint)
        parts = [
            f"<LPStandardForm: max c^T x | A * x = b ({m} rows), x >= 0 | n = {N}>",
            _arr_str(self.A, "A: "),
            _arr_str(self.b, "b: "),
            _arr_str(self.c, "c: "),
            f"var_names: {self.var_names[:5]}{' ...' if vn>5 else ''}",
            # f"base_indices_hint: {hint}",
        ]
        return "\n".join(parts)


@dataclass
class StandardFormTransform:
    """
    从原问题到标准形的变换信息（用于解与目标值的还原）。
    - kind[j]: 'shift' | 'split' | 'fixed'
    - shift[j]: 若 'shift'，原变量 x_j = y_k + shift[j]
    - keep_index[j]: 若 'shift'，y 的列索引 k
    - split_index[j]: 若 'split'，(k_pos, k_neg)
    - fixed_value[j]: 若 'fixed'，常数值
    - z0: 由平移/固定引入的目标常数项偏移
    - flipped_to_max: 是否做了 min→max 的翻号（本函数为 True）
    """
    kind: List[str]
    shift: List[float]
    keep_index: List[Optional[int]]
    split_index: List[Optional[Tuple[int, int]]]
    fixed_value: List[Optional[float]]
    z0: float
    flipped_to_max: bool = True

    def __str__(self) -> str:
        n = len(self.kind)
        # 统计各类变量数量
        n_fixed = sum(1 for k in self.kind if k == "fixed")
        n_shift = sum(1 for k in self.kind if k == "shift")
        n_split = sum(1 for k in self.kind if k == "split")
        # 展示前几个映射样例
        samples = []
        shown = 0
        for j, k in enumerate(self.kind):
            if shown >= 5:
                break
            if k == "fixed":
                samples.append(f"x[{j}] = {self.fixed_value[j]}")
            elif k == "shift":
                samples.append(f"x[{j}] = y[{self.keep_index[j]}] + {self.shift[j]}")
            else:  # split
                kp, kn = (self.split_index[j] or (None, None))
                samples.append(f"x[{j}] = y+[{kp}] - y-[{kn}]")
            shown += 1
        sample_line = "; ".join(samples)
        return (
            f"<StandardFormTransform n={n} z0={self.z0:.10g} flipped_to_max={self.flipped_to_max}>\n"
            f"kinds: fixed={n_fixed}, shift={n_shift}, split={n_split}\n"
            f"examples: {sample_line}"
        )

@dataclass
class Tableau:
    """
    单纯形表（simplex tableau）。
    约定：
      - A (m, n): 约束系数矩阵（标准形：Ax = b, x >= 0）
      - b (m,):   右端项
      - c (n,):   目标行（约化成本；对 max：最优时应满足 c <= 0）
      - z:        当前目标值（标准形 max 的 z）
      - basis:    基变量列索引（长度 m）
      - nonbasis: 非基变量列索引（长度 n - m）
      - var_names:（可选）变量名，用于调试/打印
    """
    A: Array
    b: Array
    c: Array          # reduced costs (objective row)
    z: float
    basis: List[int]
    nonbasis: List[int]
    iteration: int = 0
    var_names: Optional[List[str]] = None

    # 便捷属性
    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    def copy(self) -> "Tableau":
        return Tableau(
            A=self.A.copy(),
            b=self.b.copy(),
            c=self.c.copy(),
            z=float(self.z),
            basis=list(self.basis),
            nonbasis=list(self.nonbasis),
            iteration=int(self.iteration),
            var_names=None if self.var_names is None else list(self.var_names),
        )

    def __str__(self) -> str:
        head = (
            f"<Tableau it={self.iteration} | m={self.m}, n={self.n} | "
            f"z={self.z:.10g} | basis={self._list_head(self.basis)} "
            f"| nonbasis={self._list_head(self.nonbasis)}>"
        )
        A_str = _arr_str(self.A, "A: ")
        b_str = _arr_str(self.b, "b: ")
        c_str = _arr_str(self.c, "c(reduced): ")
        names = ""
        if self.var_names:
            preview = ", ".join(self.var_names[:6]) + (" ..." if len(self.var_names) > 6 else "")
            names = f"\nvar_names[0:6]: [{preview}]"
        return "\n".join([head, A_str, b_str, c_str]) + names

    @staticmethod
    def _list_head(xs: List[int], k: int = 8) -> str:
        if not xs:
            return "[]"
        if len(xs) <= k:
            return "[" + ", ".join(map(str, xs)) + "]"
        return "[" + ", ".join(map(str, xs[:k])) + f", ... x{len(xs)-k} more]"
    
@dataclass
class LPSolution:
    success: bool
    status: StatusType
    fun: float
    x: Array
    nit: int

    def __str__(self) -> str:
        x_str = _arr_str(self.x, "x: ")
        return (
            f"<LPSolution: success={self.success} | status={self.status} | nit={self.nit}>\n"
            f"fun: {self.fun:.10g}\n{x_str}"
        )