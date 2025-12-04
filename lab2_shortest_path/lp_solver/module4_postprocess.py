from typing import Literal
import numpy as np
from .types import StandardFormTransform, LPSolution
from .exceptions import InvalidInputError

def recover_primal(x_std: np.ndarray, xfm: StandardFormTransform) -> np.ndarray:
    """
    从标准形变量 x_std 还原回原问题变量 x。
    支持：
      - kind == "fixed": x_j = fixed_value[j]
      - kind == "shift": x_j = y_k + shift[j]，其中 k = keep_index[j]
      - kind == "split": x_j = y_pos[kp] - y_neg[kn]，其中 (kp,kn) = split_index[j]
    """
    n = len(xfm.kind)
    x = np.zeros(n, dtype=float)
    for j in range(n):
        kj = xfm.keep_index[j]
        kind = xfm.kind[j]
        if kind == "fixed":
            v = xfm.fixed_value[j]
            x[j] = 0.0 if v is None else float(v)
        elif kind == "shift":
            base = 0.0 if kj is None else float(x_std[kj])
            x[j] = base + float(xfm.shift[j])
        elif kind == "split":
            t = xfm.split_index[j]
            kp, kn = (None, None) if t is None else t
            yp = 0.0 if kp is None else float(x_std[kp])
            yn = 0.0 if kn is None else float(x_std[kn])
            x[j] = yp - yn
        else:
            raise InvalidInputError(f"Unknown transform kind: {kind}", context={"index": j})
    return x

def recover_objective(z_std: float, xfm: StandardFormTransform, sense: Literal["min","max"]="min") -> float:
    # 标准形我们按 max 求解，原问题通常是 min
    if sense == "min":
        return -float(z_std) + float(xfm.z0)
    else:
        # 若原问题本来就是 max，则不要翻号；仅加上常数项
        return float(z_std) + float(xfm.z0)

def recover_solution(std_sol: LPSolution,
                     xfm: StandardFormTransform,
                     sense: Literal["min","max"]="min") -> LPSolution:
    """
    将“标准形(max)解”还原成“原问题解”，并以 LPSolution 返回。
    - 复制 success/status/nit
    - x: 从 x_std → x_orig
    - fun: 从 z_std → 原问题目标值（min: -z+z0；max: z+z0）
    """
    x_orig = recover_primal(std_sol.x, xfm)
    f_orig = recover_objective(std_sol.fun, xfm, sense=sense)
    return LPSolution(
        success=std_sol.success,
        status=std_sol.status,
        fun=f_orig,
        x=x_orig,
        nit=std_sol.nit,
    )
