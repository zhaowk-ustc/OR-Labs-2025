"""正确性检查（占位）

- 可行性检查
- 最优性检查
- 循环检测
"""

from .types import LPProblem, LPSolution


def check_feasible(problem: LPProblem) -> bool:
    return True

def check_optimal(solution: LPSolution) -> bool:
    return False

def detect_cycle(history) -> bool:
    return False
