"""
文件类工具：read_file / write_file / edit_file / list_dir。

四个工具全部是纯本地实现（标准库 `pathlib` / `os`），没有调用任何
模型厂商 API 侧托管的"文件工具"（比如 OpenAI 的 Files API）。
"""
from __future__ import annotations

import os
from pathlib import Path

from ..config import AgentConfig
from .base import Tool, ToolResult
from .sandbox import PathEscapeError, resolve_safe_path


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2:]
    return f"{head}\n\n... [内容过长，已截断 {len(text) - limit} 个字符] ...\n\n{tail}"


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------
def _read_file(tool_input: dict, config: AgentConfig) -> ToolResult:
    raw_path = tool_input.get("path", "")
    try:
        path = resolve_safe_path(raw_path, config.workspace_dir)
    except PathEscapeError as e:
        return ToolResult(output=str(e), is_error=True)

    if not path.exists():
        return ToolResult(output=f"文件不存在: {raw_path}", is_error=True)
    if path.is_dir():
        return ToolResult(output=f"'{raw_path}' 是一个目录，不是文件，请改用 list_dir。", is_error=True)

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001 - 工具层需要把任何异常都转换成可读的错误信息
        return ToolResult(output=f"读取文件失败: {e}", is_error=True)

    numbered = "\n".join(f"{i + 1:>5}\t{line}" for i, line in enumerate(content.splitlines()))
    return ToolResult(output=_truncate(numbered, config.max_file_read_chars))


READ_FILE_TOOL = Tool(
    name="read_file",
    description=(
        "读取工作目录内某个文本文件的内容，返回带行号的文本（行号从 1 开始），"
        "行号是为了方便后续用 edit_file 精确定位要修改的内容。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对于工作目录的文件路径，例如 'src/main.py'"},
        },
        "required": ["path"],
    },
    handler=_read_file,
)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------
def _write_file(tool_input: dict, config: AgentConfig) -> ToolResult:
    raw_path = tool_input.get("path", "")
    content = tool_input.get("content", "")
    try:
        path = resolve_safe_path(raw_path, config.workspace_dir)
    except PathEscapeError as e:
        return ToolResult(output=str(e), is_error=True)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return ToolResult(output=f"写入文件失败: {e}", is_error=True)

    return ToolResult(output=f"已写入 {raw_path}（{len(content)} 个字符）。")


WRITE_FILE_TOOL = Tool(
    name="write_file",
    description=(
        "创建新文件或整体覆盖写入已有文件。适合创建新文件，或对已有文件做大范围重写。"
        "如果只是想修改文件中的一小部分，请优先使用 edit_file，避免误删无关内容。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对于工作目录的文件路径"},
            "content": {"type": "string", "description": "要写入的完整文件内容"},
        },
        "required": ["path", "content"],
    },
    handler=_write_file,
)


# ---------------------------------------------------------------------------
# edit_file（基于精确字符串替换，类似 sed 的一次性、可核验替换）
# ---------------------------------------------------------------------------
def _edit_file(tool_input: dict, config: AgentConfig) -> ToolResult:
    raw_path = tool_input.get("path", "")
    old_str = tool_input.get("old_str", "")
    new_str = tool_input.get("new_str", "")

    if old_str == "":
        return ToolResult(output="old_str 不能为空字符串，请提供要被替换的原始内容。", is_error=True)

    try:
        path = resolve_safe_path(raw_path, config.workspace_dir)
    except PathEscapeError as e:
        return ToolResult(output=str(e), is_error=True)

    if not path.exists():
        return ToolResult(output=f"文件不存在: {raw_path}，如需新建文件请使用 write_file。", is_error=True)

    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return ToolResult(output=f"读取文件失败: {e}", is_error=True)

    occurrences = original.count(old_str)
    if occurrences == 0:
        return ToolResult(
            output=(
                "未在文件中找到 old_str 指定的内容，替换失败。"
                "请先用 read_file 重新确认文件当前的准确内容（注意空格、缩进、换行是否完全一致）。"
            ),
            is_error=True,
        )
    if occurrences > 1:
        return ToolResult(
            output=(
                f"old_str 在文件中出现了 {occurrences} 次，为避免改错地方已拒绝执行。"
                "请提供更长、包含上下文、能唯一定位目标位置的 old_str。"
            ),
            is_error=True,
        )

    updated = original.replace(old_str, new_str, 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return ToolResult(output=f"写入文件失败: {e}", is_error=True)

    return ToolResult(output=f"已成功替换 '{raw_path}' 中的 1 处内容。")


EDIT_FILE_TOOL = Tool(
    name="edit_file",
    description=(
        "对已有文件做精确的局部编辑：把文件中唯一出现的 old_str 替换为 new_str。"
        "old_str 必须与文件中的内容逐字符完全匹配（包括缩进、空格），且在文件中只能出现一次，"
        "否则会拒绝执行并返回原因，请据此调整后重试。适合小范围、可核验的修改。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对于工作目录的文件路径"},
            "old_str": {"type": "string", "description": "要被替换的原始内容，必须唯一匹配"},
            "new_str": {"type": "string", "description": "替换后的新内容"},
        },
        "required": ["path", "old_str", "new_str"],
    },
    handler=_edit_file,
)


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------
def _list_dir(tool_input: dict, config: AgentConfig) -> ToolResult:
    raw_path = tool_input.get("path", ".")
    try:
        path = resolve_safe_path(raw_path, config.workspace_dir)
    except PathEscapeError as e:
        return ToolResult(output=str(e), is_error=True)

    if not path.exists():
        return ToolResult(output=f"路径不存在: {raw_path}", is_error=True)
    if not path.is_dir():
        return ToolResult(output=f"'{raw_path}' 不是目录。", is_error=True)

    lines: list[str] = []
    for root, dirnames, filenames in os.walk(path):
        # 跳过常见的噪声目录，避免把 .git / node_modules 等塞进上下文
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "node_modules", ".venv"}]
        rel_root = Path(root).relative_to(path)
        depth = len(rel_root.parts)
        indent = "  " * depth
        if str(rel_root) != ".":
            lines.append(f"{indent}{rel_root.name}/")
        for f in sorted(filenames):
            lines.append(f"{indent}  {f}")

    if not lines:
        return ToolResult(output="(空目录)")
    return ToolResult(output=_truncate("\n".join(lines), config.max_tool_output_chars))


LIST_DIR_TOOL = Tool(
    name="list_dir",
    description="递归列出工作目录内某个目录的文件与子目录结构（自动跳过 .git、__pycache__ 等噪声目录）。",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对于工作目录的目录路径，默认为工作目录本身 '.'"},
        },
        "required": [],
    },
    handler=_list_dir,
)
