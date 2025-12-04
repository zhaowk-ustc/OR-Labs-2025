"""LP Solver package

包初始化文件。后续可在此引入常用符号。
"""

from . import solver as solver
from .types import LPProblem, LPSolution

__all__ = [
    "solver",
    "LPProblem",
    "LPSolution"
]
