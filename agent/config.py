"""
Mini Coding Agent 全局配置。

所有可调参数统一从环境变量 / .env 文件读取。

API Key 不允许硬编码进源码，也不允许提交到 Git 仓库。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 无论用户从哪里启动 Python，都明确加载项目根目录的 .env。
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 环境变量辅助函数
# ============================================================


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量。"""

    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"环境变量 {name} 必须是整数，当前值为 {value!r}"
        ) from exc


def _env_float(name: str, default: float) -> float:
    """读取浮点数环境变量。"""

    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"环境变量 {name} 必须是数字，当前值为 {value!r}"
        ) from exc


# ============================================================
# Agent 配置
# ============================================================


@dataclass
class AgentConfig:
    """
    Coding Agent 的统一配置。

    模型 API 通过 httpx 直接请求 OpenRouter REST API。

    本项目没有使用：
    - LangChain
    - AutoGen
    - OpenAI Agents SDK
    - Claude Agent SDK
    - 其它 Agent 框架

    Agent 主循环、上下文管理、工具执行、工具结果回传、
    模型输出解析、循环终止条件均由本项目自行实现。
    """

    # --------------------------------------------------------
    # 模型 API
    # --------------------------------------------------------

    api_key: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_API_KEY",
            "",
        ).strip()
    )

    api_base: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_API_BASE",
            "https://openrouter.ai/api/v1",
        ).strip().rstrip("/")
    )

    model: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_MODEL",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        ).strip()
    )

    api_timeout: float = field(
        default_factory=lambda: _env_float(
            "AGENT_API_TIMEOUT",
            180.0,
        )
    )

    max_tokens: int = field(
        default_factory=lambda: _env_int(
            "AGENT_MAX_TOKENS",
            8192,
        )
    )

    # --------------------------------------------------------
    # Agent 主循环
    # --------------------------------------------------------

    max_iterations: int = field(
        default_factory=lambda: _env_int(
            "AGENT_MAX_ITERATIONS",
            40,
        )
    )

    # --------------------------------------------------------
    # 本地工作目录 / 工具
    # --------------------------------------------------------

    workspace_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "workspace"
    )

    bash_timeout: int = field(
        default_factory=lambda: _env_int(
            "AGENT_BASH_TIMEOUT",
            30,
        )
    )

    max_tool_output_chars: int = field(
        default_factory=lambda: _env_int(
            "AGENT_MAX_TOOL_OUTPUT_CHARS",
            20000,
        )
    )

    max_file_read_chars: int = field(
        default_factory=lambda: _env_int(
            "AGENT_MAX_FILE_READ_CHARS",
            30000,
        )
    )

    # --------------------------------------------------------
    # 上下文管理
    # --------------------------------------------------------

    max_context_tokens: int = field(
        default_factory=lambda: _env_int(
            "AGENT_MAX_CONTEXT_TOKENS",
            120000,
        )
    )

    keep_recent_turns: int = field(
        default_factory=lambda: _env_int(
            "AGENT_KEEP_RECENT_TURNS",
            6,
        )
    )

    approx_chars_per_token: float = field(
        default_factory=lambda: _env_float(
            "AGENT_APPROX_CHARS_PER_TOKEN",
            4.0,
        )
    )

    # --------------------------------------------------------
    # API 重试
    # --------------------------------------------------------

    max_retries: int = field(
        default_factory=lambda: _env_int(
            "AGENT_MAX_RETRIES",
            5,
        )
    )

    retry_base_delay: float = field(
        default_factory=lambda: _env_float(
            "AGENT_RETRY_BASE_DELAY",
            1.0,
        )
    )

    # --------------------------------------------------------
    # 兼容旧代码
    # --------------------------------------------------------

    @property
    def max_output_tokens(self) -> int:
        """
        如果旧代码引用 config.max_output_tokens，
        继续兼容。
        """
        return self.max_tokens

    # --------------------------------------------------------
    # 配置校验
    # --------------------------------------------------------

    def validate(self) -> None:
        """Agent 启动时尽早检查配置错误。"""

        if not self.api_key:
            raise ValueError(
                "未找到 OPENROUTER_API_KEY。\n"
                "请在项目根目录创建 .env 文件，并填写：\n"
                "OPENROUTER_API_KEY=你的_OpenRouter_API_Key"
            )

        if not self.api_base:
            raise ValueError(
                "AGENT_API_BASE 不能为空。"
            )

        if not self.model:
            raise ValueError(
                "AGENT_MODEL 不能为空。"
            )

        if self.api_timeout <= 0:
            raise ValueError(
                "AGENT_API_TIMEOUT 必须大于 0。"
            )

        if self.max_tokens <= 0:
            raise ValueError(
                "AGENT_MAX_TOKENS 必须大于 0。"
            )

        if self.max_iterations <= 0:
            raise ValueError(
                "AGENT_MAX_ITERATIONS 必须大于 0。"
            )

        if self.bash_timeout <= 0:
            raise ValueError(
                "AGENT_BASH_TIMEOUT 必须大于 0。"
            )

        if self.max_tool_output_chars <= 0:
            raise ValueError(
                "AGENT_MAX_TOOL_OUTPUT_CHARS 必须大于 0。"
            )

        if self.max_file_read_chars <= 0:
            raise ValueError(
                "AGENT_MAX_FILE_READ_CHARS 必须大于 0。"
            )

        if self.max_context_tokens <= 0:
            raise ValueError(
                "AGENT_MAX_CONTEXT_TOKENS 必须大于 0。"
            )

        if self.keep_recent_turns < 0:
            raise ValueError(
                "AGENT_KEEP_RECENT_TURNS 不能小于 0。"
            )

        if self.approx_chars_per_token <= 0:
            raise ValueError(
                "AGENT_APPROX_CHARS_PER_TOKEN 必须大于 0。"
            )

        if self.max_retries < 0:
            raise ValueError(
                "AGENT_MAX_RETRIES 不能小于 0。"
            )

        if self.retry_base_delay < 0:
            raise ValueError(
                "AGENT_RETRY_BASE_DELAY 不能小于 0。"
            )

        # workspace 不存在时自动创建。
        self.workspace_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


CONFIG = AgentConfig()
