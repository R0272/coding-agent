"""
工具系统的基础数据结构。

本项目没有使用任何 Agent 框架提供的工具装饰器。

一个工具由以下四部分组成：

1. 工具名称；
2. 给模型看的自然语言描述；
3. 描述参数格式的 JSON Schema；
4. 真正在本地执行操作的 Python 函数。

模型只能通过原生 tool calling 表达“希望调用哪个工具、传什么参数”。
真正的文件读写、代码修改和命令执行始终发生在本机，
由本项目自己的 Python 代码完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..config import AgentConfig


@dataclass
class ToolResult:
    """工具执行的统一返回结果。"""

    output: str
    is_error: bool = False


@dataclass
class Tool:
    """
    一个可供模型请求调用的本地工具。
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    # handler 接收：
    #   (工具参数字典, 全局配置)
    #
    # 返回：
    #   ToolResult
    handler: Callable[
        [dict[str, Any], AgentConfig],
        ToolResult,
    ]

    def to_tool_schema(self) -> dict[str, Any]:
        """
        导出本项目内部统一的工具定义。

        llm_client.py 会负责把这份厂商无关的定义，
        转换成 OpenRouter REST API 所需的 tool schema。
        """

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
