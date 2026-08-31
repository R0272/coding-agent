"""
路径安全校验。

所有文件类工具收到的 `path` 参数都被视为"相对于工作目录（workspace_dir）"，
并在真正操作文件之前做一次"越权访问"检查——防止模型（无论是被误导还是产生幻觉）
用 `../../etc/passwd` 这样的路径跑到工作目录之外。这是本项目自己实现的安全边界，
不依赖任何第三方沙箱产品。
"""
from __future__ import annotations

from pathlib import Path


class PathEscapeError(Exception):
    """当请求的路径试图跳出工作目录时抛出。"""


def resolve_safe_path(raw_path: str, workspace_dir: Path) -> Path:
    workspace_dir = workspace_dir.resolve()
    candidate = (workspace_dir / raw_path).resolve() if not Path(raw_path).is_absolute() \
        else Path(raw_path).resolve()

    try:
        candidate.relative_to(workspace_dir)
    except ValueError:
        raise PathEscapeError(
            f"路径 '{raw_path}' 解析后为 '{candidate}'，超出了工作目录 '{workspace_dir}' 的范围，已拒绝执行。"
        )
    return candidate
