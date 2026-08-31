"""
对话历史管理模块。

这里自行实现两件 Agent 框架通常会代劳的工作：

1. 消息拼装

   本项目内部使用统一 block 格式组织多轮历史：

       text
       tool_use
       tool_result

   llm_client.py 在真正调用模型 API 时，
   再把这些内部结构转换成 OpenRouter REST API 的消息格式。

2. 上下文长度控制

   工具输出可能包含大量代码或命令日志，
   如果历史无限增长最终会超过模型上下文窗口。

   本项目使用一个简单可解释的策略：

       估算 token 数
       ↓
       超过阈值
       ↓
       将较旧的大型 tool_result 替换为占位符
       ↓
       保留最近若干轮完整内容

   这样既减少上下文体积，也保留工具调用与工具结果之间的结构关系。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import AgentConfig


PLACEHOLDER = (
    "[此工具结果因上下文过长已被自动压缩，"
    "原始内容不再可见]"
)


@dataclass
class ConversationManager:
    """维护 Agent 的完整多轮对话历史。"""

    system_prompt: str
    config: AgentConfig
    messages: list[dict[str, Any]] = field(
        default_factory=list
    )

    # ========================================================
    # 写入
    # ========================================================

    def add_user_text(
        self,
        text: str,
    ) -> None:
        """加入一条普通用户消息。"""

        self.messages.append(
            {
                "role": "user",
                "content": text,
            }
        )

    def add_assistant_blocks(
        self,
        blocks: list[dict[str, Any]],
    ) -> None:
        """
        保存模型返回的统一 block：

            text
            tool_use
        """

        self.messages.append(
            {
                "role": "assistant",
                "content": blocks,
            }
        )

    def add_tool_results(
        self,
        results: list[dict[str, Any]],
    ) -> None:
        """
        保存一批本地工具执行结果。

        每个 tool_result 都通过 tool_use_id，
        与前一轮模型产生的 tool_use 一一对应。

        llm_client.py 会在真正请求 API 时，
        将这种内部格式转换为模型接口所要求的 tool 消息。
        """

        self.messages.append(
            {
                "role": "user",
                "content": results,
            }
        )

    # ========================================================
    # 读取
    # ========================================================

    def get_messages(
        self,
    ) -> list[dict[str, Any]]:
        """返回当前完整历史。"""

        return self.messages

    # ========================================================
    # 上下文长度估算
    # ========================================================

    def estimate_tokens(
        self,
    ) -> int:
        """
        使用字符数进行近似 token 估算。

        这不是精确 tokenizer，
        但实现简单、无需依赖具体模型。
        """

        raw = json.dumps(
            self.messages,
            ensure_ascii=False,
        )

        return int(
            len(raw)
            / self.config.approx_chars_per_token
        )

    # ========================================================
    # 上下文压缩
    # ========================================================

    def compact_if_needed(
        self,
        force: bool = False,
    ) -> bool:
        """
        压缩较旧的大型工具结果。

        force=False：
            只有超过上下文阈值时才执行。

        force=True：
            忽略阈值直接尝试压缩。

        返回 True：
            至少有一个 tool_result 被实际压缩。
        """

        if (
            not force
            and self.estimate_tokens()
            <= self.config.max_context_tokens
        ):
            return False

        # ----------------------------------------------------
        # 找出“最近 keep_recent_turns 个 user 轮次”
        # 的起始位置。
        # ----------------------------------------------------

        cutoff = len(
            self.messages
        )

        user_turns_seen = 0

        for index in range(
            len(self.messages) - 1,
            -1,
            -1,
        ):

            if (
                self.messages[index]["role"]
                == "user"
            ):
                user_turns_seen += 1

            if (
                user_turns_seen
                > self.config.keep_recent_turns
            ):
                cutoff = index + 1
                break

        else:
            cutoff = 0

        compacted_any = False

        # ----------------------------------------------------
        # 只压缩 cutoff 之前的旧 tool_result。
        #
        # 不删除整个消息，以保留：
        #
        # tool_use
        #    ↕
        # tool_result
        #
        # 之间的对应结构。
        # ----------------------------------------------------

        for message in self.messages[:cutoff]:

            if (
                message["role"] != "user"
                or not isinstance(
                    message["content"],
                    list,
                )
            ):
                continue

            for block in message["content"]:

                if not (
                    isinstance(block, dict)
                    and block.get("type")
                    == "tool_result"
                ):
                    continue

                content = block.get(
                    "content"
                )

                if isinstance(
                    content,
                    str,
                ):
                    text = content
                else:
                    text = json.dumps(
                        content,
                        ensure_ascii=False,
                    )

                # 很短的结果没必要压缩。
                if (
                    len(text) > 200
                    and text != PLACEHOLDER
                ):
                    block["content"] = (
                        PLACEHOLDER
                    )

                    compacted_any = True

        return compacted_any
