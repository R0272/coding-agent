"""
大模型 API 访问层。

本项目不使用任何模型厂商 SDK，也不使用任何 Agent SDK。

这里只使用普通 HTTP 库 httpx，直接调用 OpenRouter 提供的
OpenAI-compatible REST API：

    POST /api/v1/chat/completions

本项目自行实现：

- 对话历史管理
- 上下文压缩
- 工具定义
- 工具本地执行
- tool calling 参数解析
- 工具结果回传
- Agent 主循环
- 循环终止条件
- 错误处理与重试

llm_client.py 的职责仅包括：

1. 将项目内部消息转换成 REST API 请求格式；
2. 使用 httpx 发送 HTTP 请求；
3. 将 OpenRouter 响应转换成项目内部统一格式；
4. 对网络异常、限流、服务端错误和异常响应进行有限重试。
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .config import AgentConfig


# ============================================================
# 异常
# ============================================================


class ModelAPIError(RuntimeError):
    """模型 API 请求失败。"""


# ============================================================
# 项目内部统一响应结构
#
# core.py 只依赖这些结构，不需要了解 OpenRouter 的 JSON。
# ============================================================


@dataclass
class TextBlock:
    type: str
    text: str


@dataclass
class ToolUseBlock:
    type: str
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class NormalizedResponse:
    content: list[Any]
    stop_reason: str | None


# ============================================================
# LLM Client
# ============================================================


class LLMClient:
    """
    OpenRouter REST API 的薄封装。

    这里不执行任何 Agent 工具。
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

        self.endpoint = (
            config.api_base.rstrip("/")
            + "/chat/completions"
        )

        self.client = httpx.Client(
            timeout=httpx.Timeout(
                timeout=config.api_timeout,
                connect=min(
                    20.0,
                    config.api_timeout,
                ),
            )
        )

    # ========================================================
    # 对外接口
    # ========================================================

    def create_message(
        self,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """
        调用模型。

        同时兼容：

            system=...

        和旧代码中的：

            system_prompt=...
        """

        system_prompt = kwargs.pop(
            "system_prompt",
            None,
        )

        if system_prompt is not None:
            system = system_prompt

        if kwargs:
            unknown = ", ".join(
                sorted(kwargs.keys())
            )

            raise TypeError(
                f"create_message 收到了未知参数: {unknown}"
            )

        system = system or ""
        messages = messages or []
        tools = tools or []

        api_messages = self._convert_messages(
            system=system,
            messages=messages,
        )

        api_tools = self._convert_tools(
            tools
        )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": self.config.max_tokens,
        }

        if api_tools:
            payload["tools"] = api_tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": (
                f"Bearer {self.config.api_key}"
            ),
            "Content-Type": "application/json",
            "X-Title": "Mini Coding Agent",

            # 出问题时让 OpenRouter 返回更多路由信息，
            # 方便调试，不包含用户的 API Key。
            "X-OpenRouter-Metadata": "enabled",
        }

        return self._request_with_retry(
            headers=headers,
            payload=payload,
        )

    # ========================================================
    # HTTP 请求 + 自动重试
    # ========================================================

    def _request_with_retry(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> NormalizedResponse:
        """
        发送请求并处理重试。

        自动重试：

        - 网络连接错误
        - 请求超时
        - HTTP 408
        - HTTP 409
        - HTTP 429
        - HTTP 5xx
        - HTTP 200 但返回异常 JSON
        - HTTP 200 但没有 choices
        - HTTP 200 但 message 格式异常

        不自动重试：

        - 400 请求参数错误
        - 401 Key 错误
        - 403 权限问题
        - 404 模型不存在
        """

        total_attempts = (
            self.config.max_retries + 1
        )

        last_error: Exception | None = None

        for attempt in range(
            total_attempts
        ):
            # ------------------------------------------------
            # 发送 HTTP 请求
            # ------------------------------------------------

            try:
                response = self.client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                )

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as exc:
                last_error = exc

                if (
                    attempt
                    >= total_attempts - 1
                ):
                    raise ModelAPIError(
                        "模型 API 网络请求失败，"
                        f"已重试 {self.config.max_retries} 次："
                        f"{exc}"
                    ) from exc

                self._sleep_before_retry(
                    attempt
                )

                continue

            # ------------------------------------------------
            # 尝试解析 JSON
            # ------------------------------------------------

            try:
                data = response.json()

            except ValueError as exc:
                last_error = exc

                # 即使 HTTP 是 200，
                # 如果服务器返回了损坏的内容，
                # 也把它视作临时服务异常重试。
                if (
                    response.status_code
                    < 500
                    and not (
                        200
                        <= response.status_code
                        < 300
                    )
                ):
                    raise ModelAPIError(
                        "模型 API 返回了非 JSON 内容："
                        f"HTTP {response.status_code}: "
                        f"{response.text[:1000]}"
                    ) from exc

                if (
                    attempt
                    >= total_attempts - 1
                ):
                    raise ModelAPIError(
                        "模型 API 多次返回无法解析的 JSON。"
                    ) from exc

                self._sleep_before_retry(
                    attempt
                )

                continue

            # ------------------------------------------------
            # HTTP 成功
            # ------------------------------------------------

            if (
                200
                <= response.status_code
                < 300
            ):
                # --------------------------------------------
                # 某些网关 / provider 异常情况下，
                # HTTP 状态可能成功，但 JSON 里仍带 error。
                # --------------------------------------------

                embedded_error = data.get(
                    "error"
                )

                if embedded_error:
                    error_text = (
                        self._extract_error_from_data(
                            data
                        )
                    )

                    error_code = (
                        self._extract_error_code(
                            data
                        )
                    )

                    # 这类通常属于临时 provider 问题。
                    if self._is_retryable_code(
                        error_code
                    ):
                        last_error = ModelAPIError(
                            error_text
                        )

                        if (
                            attempt
                            >= total_attempts - 1
                        ):
                            raise ModelAPIError(
                                "模型服务异常且重试耗尽："
                                f"{error_text}"
                            )

                        self._sleep_before_retry(
                            attempt
                        )

                        continue

                    raise ModelAPIError(
                        "模型 API 返回错误："
                        f"{error_text}"
                    )

                # --------------------------------------------
                # 正常响应解析。
                #
                # 如果 HTTP 200 但缺 choices，
                # 也视作 provider 临时异常并重试。
                # --------------------------------------------

                try:
                    return self._normalize_response(
                        data
                    )

                except ModelAPIError as exc:
                    last_error = exc

                    if (
                        attempt
                        >= total_attempts - 1
                    ):
                        raise ModelAPIError(
                            "模型连续返回异常响应，"
                            "已达到最大重试次数："
                            f"{exc}"
                        ) from exc

                    self._sleep_before_retry(
                        attempt
                    )

                    continue

            # ------------------------------------------------
            # HTTP 非 2xx
            # ------------------------------------------------

            error_text = (
                self._extract_error_from_data(
                    data
                )
            )

            status_code = (
                response.status_code
            )

            # --------------------------------------------
            # 临时错误
            # --------------------------------------------

            if self._is_retryable_code(
                status_code
            ):
                last_error = ModelAPIError(
                    f"HTTP {status_code}: "
                    f"{error_text}"
                )

                if (
                    attempt
                    >= total_attempts - 1
                ):
                    raise ModelAPIError(
                        "模型 API 临时错误且重试耗尽："
                        f"HTTP {status_code}: "
                        f"{error_text}"
                    )

                self._sleep_before_retry(
                    attempt
                )

                continue

            # --------------------------------------------
            # 不可通过原样重试解决的问题
            # --------------------------------------------

            if status_code == 400:
                raise ModelAPIError(
                    "模型 API 请求格式错误："
                    f"{error_text}"
                )

            if status_code == 401:
                raise ModelAPIError(
                    "OpenRouter API Key 无效或已经失效。"
                )

            if status_code == 402:
                raise ModelAPIError(
                    "当前请求需要付费额度："
                    f"{error_text}"
                )

            if status_code == 403:
                raise ModelAPIError(
                    "OpenRouter 拒绝访问："
                    f"{error_text}"
                )

            if status_code == 404:
                raise ModelAPIError(
                    "模型或 API 地址不存在："
                    f"{error_text}"
                )

            raise ModelAPIError(
                "模型 API 返回异常状态："
                f"HTTP {status_code}: "
                f"{error_text}"
            )

        # 理论上不会运行到这里。
        if last_error is not None:
            raise ModelAPIError(
                str(last_error)
            )

        raise ModelAPIError(
            "模型 API 请求失败，原因未知。"
        )

    # ========================================================
    # 判断是否值得重试
    # ========================================================

    @staticmethod
    def _is_retryable_code(
        code: Any,
    ) -> bool:
        """
        判断 HTTP / provider 错误是否适合重试。
        """

        try:
            value = int(code)
        except (
            TypeError,
            ValueError,
        ):
            # 没有明确错误码时，
            # provider 异常通常值得有限重试。
            return True

        return (
            value in {
                408,
                409,
                429,
                500,
                502,
                503,
                504,
                524,
                529,
            }
            or value >= 500
        )

    # ========================================================
    # OpenRouter 错误解析
    # ========================================================

    @staticmethod
    def _extract_error_from_data(
        data: Any,
    ) -> str:
        """
        从 OpenRouter JSON 中提取可读错误。
        """

        if not isinstance(
            data,
            dict,
        ):
            return str(data)

        error = data.get(
            "error"
        )

        if isinstance(
            error,
            dict,
        ):
            message = error.get(
                "message"
            )

            if message:
                return str(
                    message
                )

            return json.dumps(
                error,
                ensure_ascii=False,
            )

        if error:
            return str(
                error
            )

        return json.dumps(
            data,
            ensure_ascii=False,
        )[:1500]

    @staticmethod
    def _extract_error_code(
        data: Any,
    ) -> Any:
        """取得 OpenRouter error.code。"""

        if not isinstance(
            data,
            dict,
        ):
            return None

        error = data.get(
            "error"
        )

        if not isinstance(
            error,
            dict,
        ):
            return None

        return error.get(
            "code"
        )

    # ========================================================
    # 工具定义转换
    # ========================================================

    @staticmethod
    def _convert_tools(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        项目内部格式：

            {
                "name": "...",
                "description": "...",
                "input_schema": {...}
            }

        转换为 OpenAI-compatible 格式：

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """

        converted: list[dict[str, Any]] = []

        for tool in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get(
                            "description",
                            "",
                        ),
                        "parameters": tool.get(
                            "input_schema",
                            {
                                "type": "object",
                                "properties": {},
                            },
                        ),
                    },
                }
            )

        return converted

    # ========================================================
    # 历史消息转换
    # ========================================================

    def _convert_messages(
        self,
        system: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        项目内部保存：

            user
            assistant(text/tool_use)
            user(tool_result)

        API 请求转换为：

            system
            user
            assistant(tool_calls)
            tool
        """

        result: list[dict[str, Any]] = []

        if system:
            result.append(
                {
                    "role": "system",
                    "content": system,
                }
            )

        for message in messages:
            role = message.get(
                "role"
            )

            content = message.get(
                "content"
            )

            if role == "assistant":
                result.append(
                    self._convert_assistant_message(
                        content
                    )
                )

            elif role == "user":
                result.extend(
                    self._convert_user_message(
                        content
                    )
                )

            else:
                result.append(
                    {
                        "role": "user",
                        "content": str(
                            content
                        ),
                    }
                )

        return result

    # ========================================================
    # assistant 消息转换
    # ========================================================

    @staticmethod
    def _convert_assistant_message(
        content: Any,
    ) -> dict[str, Any]:

        if isinstance(
            content,
            str,
        ):
            return {
                "role": "assistant",
                "content": content,
            }

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        if isinstance(
            content,
            list,
        ):
            for block in content:
                if not isinstance(
                    block,
                    dict,
                ):
                    continue

                block_type = block.get(
                    "type"
                )

                if block_type == "text":
                    text = block.get(
                        "text",
                        "",
                    )

                    if text:
                        text_parts.append(
                            str(text)
                        )

                elif block_type == "tool_use":
                    tool_id = (
                        block.get(
                            "id"
                        )
                        or (
                            "call_"
                            + uuid.uuid4().hex
                        )
                    )

                    arguments = block.get(
                        "input",
                        {},
                    )

                    tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": block.get(
                                    "name",
                                    "",
                                ),
                                "arguments": json.dumps(
                                    arguments,
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )

        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": (
                "\n".join(
                    text_parts
                )
                if text_parts
                else None
            ),
        }

        if tool_calls:
            assistant["tool_calls"] = (
                tool_calls
            )

        return assistant

    # ========================================================
    # user / tool_result 消息转换
    # ========================================================

    @staticmethod
    def _convert_user_message(
        content: Any,
    ) -> list[dict[str, Any]]:

        if isinstance(
            content,
            str,
        ):
            return [
                {
                    "role": "user",
                    "content": content,
                }
            ]

        if not isinstance(
            content,
            list,
        ):
            return [
                {
                    "role": "user",
                    "content": str(
                        content
                    ),
                }
            ]

        result: list[dict[str, Any]] = []

        normal_text: list[str] = []

        for block in content:
            if not isinstance(
                block,
                dict,
            ):
                normal_text.append(
                    str(block)
                )

                continue

            block_type = block.get(
                "type"
            )

            if block_type == "tool_result":
                tool_content = block.get(
                    "content",
                    "",
                )

                if not isinstance(
                    tool_content,
                    str,
                ):
                    tool_content = json.dumps(
                        tool_content,
                        ensure_ascii=False,
                    )

                if block.get(
                    "is_error"
                ):
                    tool_content = (
                        "[工具执行失败]\n"
                        + tool_content
                    )

                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get(
                            "tool_use_id",
                            "",
                        ),
                        "content": tool_content,
                    }
                )

            elif block_type == "text":
                text = block.get(
                    "text",
                    "",
                )

                if text:
                    normal_text.append(
                        str(text)
                    )

        if normal_text:
            result.insert(
                0,
                {
                    "role": "user",
                    "content": "\n".join(
                        normal_text
                    ),
                },
            )

        return result

    # ========================================================
    # 模型响应标准化
    # ========================================================

    @staticmethod
    def _normalize_response(
        data: dict[str, Any],
    ) -> NormalizedResponse:
        """
        将 OpenRouter Chat Completions 响应转换为
        项目内部统一的 TextBlock / ToolUseBlock。

        Tool Calling 时：

            message.content == None

        是合法情况。
        """

        choices = data.get(
            "choices"
        )

        if not isinstance(
            choices,
            list,
        ) or not choices:
            summary = json.dumps(
                data,
                ensure_ascii=False,
            )[:1500]

            raise ModelAPIError(
                "模型响应缺少 choices。"
                f"响应摘要: {summary}"
            )

        choice = choices[0]

        if not isinstance(
            choice,
            dict,
        ):
            raise ModelAPIError(
                "模型响应 choices[0] 格式异常。"
            )

        message = choice.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):
            raise ModelAPIError(
                "模型响应缺少合法 message。"
            )

        blocks: list[Any] = []

        # ----------------------------------------------------
        # 文本
        # ----------------------------------------------------

        content = message.get(
            "content"
        )

        if isinstance(
            content,
            str,
        ):
            if content.strip():
                blocks.append(
                    TextBlock(
                        type="text",
                        text=content,
                    )
                )

        elif isinstance(
            content,
            list,
        ):
            text_parts: list[str] = []

            for part in content:
                if isinstance(
                    part,
                    str,
                ):
                    text_parts.append(
                        part
                    )

                elif isinstance(
                    part,
                    dict,
                ):
                    text = part.get(
                        "text"
                    )

                    if text:
                        text_parts.append(
                            str(text)
                        )

            if text_parts:
                blocks.append(
                    TextBlock(
                        type="text",
                        text="\n".join(
                            text_parts
                        ),
                    )
                )

        # ----------------------------------------------------
        # 工具调用
        # ----------------------------------------------------

        tool_calls = (
            message.get(
                "tool_calls"
            )
            or []
        )

        if not isinstance(
            tool_calls,
            list,
        ):
            raise ModelAPIError(
                "模型返回的 tool_calls 不是列表。"
            )

        for call in tool_calls:
            if not isinstance(
                call,
                dict,
            ):
                continue

            function = call.get(
                "function"
            )

            if not isinstance(
                function,
                dict,
            ):
                continue

            name = function.get(
                "name",
                "",
            )

            raw_arguments = function.get(
                "arguments",
                "{}",
            )

            if isinstance(
                raw_arguments,
                dict,
            ):
                arguments = (
                    raw_arguments
                )

            else:
                try:
                    arguments = json.loads(
                        raw_arguments
                        or "{}"
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):
                    arguments = {
                        "_invalid_arguments": (
                            str(
                                raw_arguments
                            )
                        )
                    }

            if not isinstance(
                arguments,
                dict,
            ):
                arguments = {
                    "_invalid_arguments": (
                        str(
                            arguments
                        )
                    )
                }

            tool_id = (
                call.get(
                    "id"
                )
                or (
                    "call_"
                    + uuid.uuid4().hex
                )
            )

            blocks.append(
                ToolUseBlock(
                    type="tool_use",
                    id=tool_id,
                    name=str(
                        name
                    ),
                    input=arguments,
                )
            )

        # ----------------------------------------------------
        # 完全空的响应不应假装任务完成
        # ----------------------------------------------------

        if not blocks:
            summary = json.dumps(
                data,
                ensure_ascii=False,
            )[:1500]

            raise ModelAPIError(
                "模型本轮没有返回文本或工具调用。"
                f"响应摘要: {summary}"
            )

        finish_reason = choice.get(
            "finish_reason"
        )

        if finish_reason == "length":
            stop_reason = (
                "max_tokens"
            )

        elif finish_reason == "tool_calls":
            stop_reason = (
                "tool_use"
            )

        else:
            stop_reason = (
                str(finish_reason)
                if finish_reason is not None
                else None
            )

        return NormalizedResponse(
            content=blocks,
            stop_reason=stop_reason,
        )

    # ========================================================
    # 指数退避
    # ========================================================

    def _sleep_before_retry(
        self,
        attempt: int,
    ) -> None:
        """
        指数退避 + 少量随机抖动。
        """

        delay = (
            self.config.retry_base_delay
            * (2 ** attempt)
        )

        jitter = random.uniform(
            0,
            max(
                delay * 0.25,
                0.01,
            ),
        )

        time.sleep(
            delay + jitter
        )

    # ========================================================
    # 资源释放
    # ========================================================

    def close(self) -> None:
        """释放 HTTP 连接池。"""

        self.client.close()
