from typing import Any, Optional, Dict

Context = Dict[str, Any]

__all__ = [
    "LPException",
    "InvalidInputError",
    "InfeasibleError",
    "UnboundedError",
    "IterationLimitExceeded",
    "TimeLimitExceeded",
    "NumericIssueError",
    "PresolveError",
]

class LPException(Exception):
    """LP 求解过程中可预期的异常基类（简化版）。"""
    def __init__(self, message: str = "", *, context: Optional[Context] = None):
        super().__init__(message)
        self.message = message
        self.context: Context = context or {}

    def __str__(self) -> str:
        # 与原生异常类似：主要打印 message；没有就打印类名
        return self.message or self.__class__.__name__

class InvalidInputError(LPException):
    """输入不合法（维度不匹配、NaN、bounds 错误等）。"""
    pass

class InfeasibleError(LPException):
    """不可行问题。"""
    pass

class UnboundedError(LPException):
    """无界问题。"""
    pass

class IterationLimitExceeded(LPException):
    """迭代次数超限。"""
    pass

class TimeLimitExceeded(LPException):
    """时间限制超限。"""
    pass

class NumericIssueError(LPException):
    """数值问题（枢轴过小、奇异、溢出等）。"""
    pass

class PresolveError(LPException):
    """预处理/标准化阶段错误。"""
    pass