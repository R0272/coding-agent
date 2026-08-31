"""
run_bash 工具：在工作目录内执行 shell 命令。

用标准库 `subprocess` 直接起子进程，设置超时与工作目录，
不依赖任何模型厂商托管的"代码执行"服务（如 Code Interpreter）。
"""
from __future__ import annotations

import subprocess

from ..config import AgentConfig
from .base import Tool, ToolResult


def _run_bash(tool_input: dict, config: AgentConfig) -> ToolResult:
    command = tool_input.get("command", "")
    if not command.strip():
        return ToolResult(output="command 不能为空。", is_error=True)

    timeout = tool_input.get("timeout") or config.bash_timeout

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(config.workspace_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            output=f"命令执行超过 {timeout} 秒被强制终止，请考虑拆分任务或增加 timeout 参数。",
            is_error=True,
        )
    except Exception as e:  # noqa: BLE001 - 任何启动子进程失败的情况都要转换成可读错误
        return ToolResult(output=f"命令启动失败: {e}", is_error=True)

    def _cut(s: str) -> str:
        if len(s) <= config.max_tool_output_chars:
            return s
        return s[: config.max_tool_output_chars] + "\n...(输出过长，已截断)"

    parts = [f"$ {command}", f"[退出码: {completed.returncode}]"]
    if completed.stdout:
        parts.append(f"--- stdout ---\n{_cut(completed.stdout)}")
    if completed.stderr:
        parts.append(f"--- stderr ---\n{_cut(completed.stderr)}")

    output = "\n".join(parts)
    # 非 0 退出码不一定是"工具本身出错"（比如 grep 找不到匹配也会返回 1），
    # 因此这里不强制标记 is_error，而是把退出码和 stderr 都如实回传给模型，
    # 交由模型自己判断下一步该怎么做——这更贴近真实开发者使用终端的方式。
    return ToolResult(output=output, is_error=False)


RUN_BASH_TOOL = Tool(
    name="run_bash",
    description=(
        "在沙箱工作目录下执行一条 shell 命令（例如运行测试、安装依赖、执行脚本、查看 git diff 等），"
        "返回标准输出、标准错误与退出码。命令有超时限制，避免长时间阻塞或死循环。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的 shell 命令"},
            "timeout": {"type": "integer", "description": "可选，超时时间（秒），默认使用全局配置"},
        },
        "required": ["command"],
    },
    handler=_run_bash,
)
