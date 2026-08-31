"""
极简终端日志/展示模块。

不引入任何第三方 UI 库，仅用 ANSI 转义序列做颜色区分，
目的是让运行过程（模型在想什么、调用了什么工具、工具返回了什么）对用户完全透明——
这对"演示视频"和"面试讲解设计"都很重要：评委需要能一眼看懂 agent 每一步在做什么。
"""
from __future__ import annotations

_RESET = "\033[0m"
_COLORS = {
    "user": "\033[36m",      # 青色：用户输入
    "assistant": "\033[32m", # 绿色：模型的文字回复
    "tool_call": "\033[33m", # 黄色：模型发起的工具调用
    "tool_result": "\033[90m",  # 灰色：工具执行结果
    "system": "\033[35m",    # 品红：系统/框架级提示
    "error": "\033[31m",     # 红色：错误
}


def _paint(tag: str, text: str) -> str:
    color = _COLORS.get(tag, "")
    return f"{color}{text}{_RESET}" if color else text


def log_system(msg: str) -> None:
    print(_paint("system", f"[系统] {msg}"))


def log_user(msg: str) -> None:
    print(_paint("user", f"\n你 > {msg}"))


def log_assistant_text(msg: str) -> None:
    if msg.strip():
        print(_paint("assistant", f"\n智能体 > {msg}"))


def log_tool_call(name: str, tool_input: dict) -> None:
    preview = str(tool_input)
    if len(preview) > 300:
        preview = preview[:300] + "...(截断)"
    print(_paint("tool_call", f"  ↳ 调用工具 [{name}]  参数: {preview}"))


def log_tool_result(name: str, result: str, is_error: bool) -> None:
    preview = result if len(result) <= 500 else result[:500] + "...(截断，完整内容已回传给模型)"
    tag = "error" if is_error else "tool_result"
    prefix = "✗ 工具出错" if is_error else "✓ 工具结果"
    print(_paint(tag, f"    {prefix} [{name}]: {preview}"))


def log_error(msg: str) -> None:
    print(_paint("error", f"[错误] {msg}"))


def log_iteration(index: int, max_iterations: int) -> None:
    print(_paint("system", f"\n——— 第 {index}/{max_iterations} 轮 ———"))
