from dataclasses import dataclass
import numpy as np
from typing import Literal

from .types import LPProblem, LPSolution, StatusType
from .module0_standardize import to_standard_form
from .module1_redundancy import prune_redundant_rows
from .module2_bigm import init_with_big_m
from .module3_simplex import iterate_simplex
from .module4_postprocess import recover_solution

from .exceptions import (
    InvalidInputError,
    InfeasibleError,
    UnboundedError,
    IterationLimitExceeded,
    NumericIssueError,
    PresolveError,
)

def _fallback_solution(status: StatusType) -> LPSolution:
    """遇到异常时构造一个统一的 LPSolution 返回值。"""
    return LPSolution(
        success=False,
        status=status,
        fun=float("nan"),
        x=np.array([], dtype=float),
        nit=0,
    )

def solve(lp: LPProblem) -> LPSolution:
    """求解线性规划问题"""
    try:
        # 模块0：标准化
        std, xfm = to_standard_form(lp)

        # 模块1：秩检查/冗余行删除
        std = prune_redundant_rows(std)

        # 模块2：大M初始化
        tab = init_with_big_m(std)

        # 模块3：单纯形迭代（最优→返回，其它抛异常）
        res_std = iterate_simplex(tab)

        # 模块4：还原到原问题（min）
        return recover_solution(res_std, xfm, sense="min")

    # ---- 语义性错误 ----
    except InfeasibleError:
        return _fallback_solution("infeasible")
    except UnboundedError:
        return _fallback_solution("unbounded")
    except IterationLimitExceeded:
        return _fallback_solution("iteration_limit")

    # ---- 预处理/数值/输入类错误 ----
    except InvalidInputError:
        return _fallback_solution("invalid_input")
    except PresolveError:
        return _fallback_solution("presolve_error")
    except NumericIssueError:
        return _fallback_solution("numeric_issue")

    # ---- 兜底：任何未预料异常 ----
    except Exception:
        return _fallback_solution("unknown_error")
