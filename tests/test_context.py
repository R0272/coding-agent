"""
测试 ConversationManager 的上下文压缩策略：
超过阈值时，较旧的大体积 tool_result 应该被替换为占位符，
但消息数量、tool_use/tool_result 的配对关系必须保持不变（否则下次请求会被 API 拒绝）。
"""
from __future__ import annotations

import unittest

from agent.config import AgentConfig
from agent.context import PLACEHOLDER, ConversationManager


class ContextCompactionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # 故意把上限设得很小，方便触发压缩
        self.config = AgentConfig(api_key="dummy", max_context_tokens=50, keep_recent_turns=1)
        self.conv = ConversationManager(system_prompt="sys", config=self.config)

    def _add_round(self, big: bool) -> None:
        big_text = "x" * 1000 if big else "short"
        self.conv.add_user_text("do something")
        self.conv.add_assistant_blocks(
            [{"type": "tool_use", "id": "id1", "name": "read_file", "input": {"path": "a"}}]
        )
        self.conv.add_tool_results(
            [{"type": "tool_result", "tool_use_id": "id1", "content": big_text, "is_error": False}]
        )

    def test_compaction_replaces_old_large_tool_results(self) -> None:
        for _ in range(4):
            self._add_round(big=True)

        message_count_before = len(self.conv.messages)
        compacted = self.conv.compact_if_needed()
        self.assertTrue(compacted)
        # 消息条数不能变，否则 tool_use/tool_result 配对会被破坏
        self.assertEqual(len(self.conv.messages), message_count_before)

        # 最旧的一轮应当已经被压缩
        oldest_tool_result = self.conv.messages[2]["content"][0]
        self.assertEqual(oldest_tool_result["content"], PLACEHOLDER)

        # 最近一轮（keep_recent_turns=1）应当原样保留
        newest_tool_result = self.conv.messages[-1]["content"][0]
        self.assertEqual(newest_tool_result["content"], "x" * 1000)

    def test_no_compaction_when_under_threshold(self) -> None:
        self.config.max_context_tokens = 10_000_000
        self._add_round(big=False)
        self.assertFalse(self.conv.compact_if_needed())


if __name__ == "__main__":
    unittest.main()
