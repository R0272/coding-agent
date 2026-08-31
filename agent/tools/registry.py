"""
工具注册表。

核心职责：

1. 维护“工具名 -> Tool 对象”的映射；
2. 向模型访问层导出所有工具的统一 JSON Schema；
3. 提供统一、带异常兜底的 execute() 入口。

工具函数即使发生未预期异常，也不会直接导致整个 Agent 进程崩溃。
异常会被转换成 ToolResult，再作为工具结果反馈给模型，
由模型决定下一步如何调整方案。
"""

from __future__ import annotations

import traceback

from ..config import AgentConfig
from .base import Tool, ToolResult
from .bash_tool import RUN_BASH_TOOL
from .file_tools import (
    EDIT_FILE_TOOL,
    LIST_DIR_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
)


class ToolRegistry:
    """保存、导出并执行本地工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具。"""

        if tool.name in self._tools:
            raise ValueError(
                f"工具名重复注册: {tool.name}"
            )

        self._tools[tool.name] = tool

    def tool_defs(self) -> list[dict]:
        """
        导出所有工具的厂商无关定义。

        llm_client.py 会在真正发送 HTTP 请求之前，
        再转换成 OpenRouter 所需的格式。
        """

        return [
            tool.to_tool_schema()
            for tool in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        tool_input: dict,
        config: AgentConfig,
    ) -> ToolResult:
        """
        执行一个模型请求的本地工具。

        任何工具内部未捕获的异常都会在这一层被转换成
        is_error=True 的 ToolResult。
        """

        tool = self._tools.get(name)

        if tool is None:
            return ToolResult(
                output=(
                    f"未知工具: {name}。"
                    f"可用工具: {list(self._tools)}"
                ),
                is_error=True,
            )

        try:
            return tool.handler(
                tool_input,
                config,
            )

        except Exception as exc:  # noqa: BLE001
            # 工具层最后一道安全网：
            # 工具执行失败不应该导致整个 Agent 退出。
            tb = traceback.format_exc(
                limit=3,
            )

            return ToolResult(
                output=(
                    f"工具 '{name}' 执行时发生未预期的异常: "
                    f"{exc}\n{tb}"
                ),
                is_error=True,
            )


def build_default_registry() -> ToolRegistry:
    """创建并注册项目默认的五个本地工具。"""

    registry = ToolRegistry()

    for tool in (
        READ_FILE_TOOL,
        WRITE_FILE_TOOL,
        EDIT_FILE_TOOL,
        LIST_DIR_TOOL,
        RUN_BASH_TOOL,
    ):
        registry.register(tool)

    return registry
