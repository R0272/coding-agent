"""
Agent 主循环 —— 本项目的核心。

一轮完整的“思考-行动”循环：

    1. 把当前对话历史 + 工具定义发送给大模型；
    2. 解析模型返回的统一响应；
    3. 展示模型的文字回复；
    4. 如果模型没有请求工具，认为任务完成；
    5. 如果模型请求工具，在本地真正执行；
    6. 把工具结果加入对话历史；
    7. 再次调用模型；
    8. 直到完成、达到最大轮数或发生不可恢复错误。

模型 REST API 的具体协议由 llm_client.py 负责适配。
"""

from __future__ import annotations

from .config import AgentConfig
from .context import ConversationManager
from .llm_client import (
    LLMClient,
    ModelAPIError,
)
from .logger import (
    log_assistant_text,
    log_error,
    log_iteration,
    log_system,
    log_tool_call,
    log_tool_result,
)
from .tools import (
    ToolRegistry,
    build_default_registry,
)


SYSTEM_PROMPT_TEMPLATE = """\
你是一个自主编程智能体（coding agent），运行在用户本机的一个沙箱工作目录中。

你的目标是帮助用户完成编程任务：
阅读代码、编写代码、修改文件、执行命令、运行测试并根据结果继续调整。

工作方式：

1. 收到任务后，如有必要先使用 list_dir / read_file 了解已有代码，
   不要凭空猜测文件内容。

2. 通过工具一步步完成任务。每次工具执行后，都要仔细查看返回结果，
   再决定下一步应该做什么。

3. 修改已有文件时优先使用 edit_file 做精确局部修改；
   新建文件或者确实需要整体重写时再使用 write_file。

4. 修改完成后，如果存在测试或者能够验证结果的命令，
   应主动使用 run_bash 实际验证，而不是假设代码一定正确。

5. 如果工具返回错误，不要机械地使用完全相同的参数重复调用。
   应根据错误内容分析原因并调整方案。

6. 当你确认任务已经完成，或者确定当前信息不足、需要用户进一步澄清时，
   直接输出最终文字总结，并且不要再调用任何工具。
   没有新的工具调用就是任务结束信号。

7. 回复使用简体中文。
   代码、命令和原始报错信息可以保持原样。

当前工作目录（沙箱根目录）：
{workspace_dir}
"""


class CodingAgent:
    """
    Coding Agent 主控制器。
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:

        self.config = config or AgentConfig()

        self.config.validate()

        self.registry = (
            registry
            or build_default_registry()
        )

        self.llm = LLMClient(
            self.config
        )

        system_prompt = (
            SYSTEM_PROMPT_TEMPLATE.format(
                workspace_dir=self.config.workspace_dir
            )
        )

        self.conversation = ConversationManager(
            system_prompt=system_prompt,
            config=self.config,
        )

    # ========================================================
    # 对外主入口
    # ========================================================

    def run_task(
        self,
        user_task: str,
    ) -> str:
        """
        接收自然语言编程任务，
        驱动 Agent 循环直到任务结束。
        """

        self.conversation.add_user_text(
            user_task
        )

        # 厂商无关的工具定义。
        # 真正转换成 OpenRouter 格式由 llm_client.py 完成。
        tool_defs = (
            self.registry.tool_defs()
        )

        for iteration in range(
            1,
            self.config.max_iterations + 1,
        ):

            log_iteration(
                iteration,
                self.config.max_iterations,
            )

            # --------------------------------------------
            # 请求模型之前检查上下文长度
            # --------------------------------------------

            if self.conversation.compact_if_needed():
                log_system(
                    "对话历史较长，已自动压缩较旧的工具输出"
                    "以节省上下文空间。"
                )

            # --------------------------------------------
            # 调用模型
            # --------------------------------------------

            try:
                response = (
                    self.llm.create_message(
                        system_prompt=(
                            self.conversation.system_prompt
                        ),
                        messages=(
                            self.conversation.get_messages()
                        ),
                        tools=tool_defs,
                    )
                )

            except ModelAPIError as exc:
                # 如果请求可能因为上下文过长等原因被拒绝，
                # 再做一次更激进的压缩并重试。
                #
                # 即使真正原因不是上下文长度，
                # 这次重试也只发生一次，不会形成死循环。
                log_error(
                    f"模型 API 请求失败：{exc}，"
                    "尝试压缩历史后重试一次..."
                )

                self._force_compact()

                try:
                    response = (
                        self.llm.create_message(
                            system_prompt=(
                                self.conversation.system_prompt
                            ),
                            messages=(
                                self.conversation.get_messages()
                            ),
                            tools=tool_defs,
                        )
                    )

                except Exception as retry_exc:  # noqa: BLE001
                    log_error(
                        "重试仍然失败，终止任务："
                        f"{retry_exc}"
                    )

                    return (
                        "任务因模型 API 请求错误被终止："
                        f"{retry_exc}"
                    )

            except Exception as exc:  # noqa: BLE001
                # 最后一层防御：
                # 理论上 llm_client 应该已经把常见 API 错误
                # 转换为 ModelAPIError。
                log_error(
                    "调用模型发生未预期异常，终止任务："
                    f"{exc}"
                )

                return (
                    "任务因模型服务错误被终止："
                    f"{exc}"
                )

            # --------------------------------------------
            # 解析模型响应
            # --------------------------------------------

            (
                text_blocks,
                tool_use_blocks,
                assistant_blocks,
            ) = self._parse_response(
                response
            )

            # 显示模型文字输出。
            for text in text_blocks:
                log_assistant_text(
                    text
                )

            # 无论是否包含工具调用，
            # 都保存 assistant 的完整响应。
            self.conversation.add_assistant_blocks(
                assistant_blocks
            )

            # ==================================================
            # 终止条件 1
            #
            # 模型输出因为 token 上限被截断。
            # 此时工具参数可能不完整，因此主动停止。
            # ==================================================

            if (
                response.stop_reason
                == "max_tokens"
            ):
                log_error(
                    "模型单次回复超出 max_tokens 被截断，"
                    "为避免基于不完整内容继续操作，任务已终止。"
                )

                return (
                    "任务终止：模型回复被截断（max_tokens），"
                    "建议增大 AGENT_MAX_TOKENS 后重试。"
                )

            # ==================================================
            # 终止条件 2
            #
            # 本轮没有任何工具调用：
            # 认为模型已经给出最终回答。
            # ==================================================

            if not tool_use_blocks:
                final_text = (
                    "\n".join(text_blocks)
                    .strip()
                )

                return (
                    final_text
                    or "(模型未返回文字内容)"
                )

            # ==================================================
            # 执行模型请求的工具
            # ==================================================

            tool_results: list[dict] = []

            for block in tool_use_blocks:

                tool_name = block["name"]
                tool_input = block["input"]

                log_tool_call(
                    tool_name,
                    tool_input,
                )

                result = (
                    self.registry.execute(
                        tool_name,
                        tool_input,
                        self.config,
                    )
                )

                log_tool_result(
                    tool_name,
                    result.output,
                    result.is_error,
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": result.output,
                        "is_error": result.is_error,
                    }
                )

            # 把所有工具执行结果作为下一轮上下文。
            self.conversation.add_tool_results(
                tool_results
            )

        # ======================================================
        # 终止条件 3
        #
        # 最大循环次数保护。
        # ======================================================

        log_error(
            "已达到最大迭代轮数"
            f"（{self.config.max_iterations}），"
            "任务被强制终止。"
        )

        return (
            "任务终止：已达到最大迭代轮数，"
            "任务可能未完全完成，请检查当前进度"
            "或提高 AGENT_MAX_ITERATIONS。"
        )

    # ========================================================
    # 模型响应解析
    # ========================================================

    @staticmethod
    def _parse_response(
        response,
    ):
        """
        解析 llm_client.py 返回的统一响应。

        llm_client 已经把不同模型 API 的原始 JSON 转换为：

            TextBlock
            ToolUseBlock

        core.py 在这里仍然自行完成：

        - 识别普通文字；
        - 识别工具调用；
        - 提取工具参数；
        - 构造可保存进对话历史的字典结构。

        因此 Agent 主循环本身完全不依赖特定模型厂商。
        """

        text_blocks: list[str] = []
        tool_use_blocks: list[dict] = []
        assistant_blocks: list[dict] = []

        # 永远把 None 当成空列表处理，
        # 防止模型供应商返回空 content 时导致进程崩溃。
        for block in (
            response.content or []
        ):

            block_type = getattr(
                block,
                "type",
                None,
            )

            if block_type == "text":

                text = getattr(
                    block,
                    "text",
                    "",
                )

                if text:
                    text_blocks.append(
                        text
                    )

                    assistant_blocks.append(
                        {
                            "type": "text",
                            "text": text,
                        }
                    )

            elif block_type == "tool_use":

                tool_id = getattr(
                    block,
                    "id",
                    "",
                )

                tool_name = getattr(
                    block,
                    "name",
                    "",
                )

                tool_input = getattr(
                    block,
                    "input",
                    {},
                )

                tool_block = {
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }

                tool_use_blocks.append(
                    tool_block
                )

                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_input,
                    }
                )

            # 未来如果模型增加 reasoning / thinking 等其它
            # block 类型，这里会安全忽略，
            # 不会因此让整个 Agent 崩溃。

        return (
            text_blocks,
            tool_use_blocks,
            assistant_blocks,
        )

    # ========================================================
    # 强制上下文压缩
    # ========================================================

    def _force_compact(
        self,
    ) -> None:
        """
        比正常压缩更激进地缩短旧历史。

        临时减少需要保留的最近轮数，
        压缩完成后恢复原配置。
        """

        original = (
            self.config.keep_recent_turns
        )

        try:
            self.config.keep_recent_turns = max(
                1,
                original // 2,
            )

            self.conversation.compact_if_needed(
                force=True
            )

        finally:
            # 即使压缩过程中发生异常，
            # 也恢复原配置。
            self.config.keep_recent_turns = (
                original
            )
