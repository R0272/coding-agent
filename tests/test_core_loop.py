"""
对 CodingAgent 主循环的集成测试。

这里用 unittest.mock 伪造 LLMClient.create_message 的返回值，
完全不需要真实网络请求或 API Key，就能验证本项目自己写的核心逻辑是否正确：
  - 模型输出（text / tool_use blocks）解析是否正确；
  - 工具调用是否被正确调度执行，结果是否正确写回对话历史；
  - "无工具调用即结束"这一终止条件是否生效；
  - 达到最大轮数时是否会被强制终止，而不是无限循环下去。
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from agent.config import AgentConfig
from agent.core import CodingAgent


@dataclass
class FakeBlock:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None


@dataclass
class FakeMessage:
    content: list[FakeBlock]
    stop_reason: str = "end_turn"


class CoreLoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="agent_core_test_"))
        self.config = AgentConfig(api_key="dummy", workspace_dir=self.tmp_dir, max_iterations=5)

    def test_stops_when_no_tool_use(self) -> None:
        """模型第一次回复就只有文字、没有工具调用 -> 应当立刻结束，返回该文字。"""
        fake_response = FakeMessage(
            content=[FakeBlock(type="text", text="任务很简单，直接回答：42。")],
            stop_reason="end_turn",
        )
        agent = CodingAgent(config=self.config)
        with patch.object(agent.llm, "create_message", return_value=fake_response) as mocked:
            result = agent.run_task("1+41 等于几？")
        mocked.assert_called_once()
        self.assertIn("42", result)

    def test_executes_tool_then_finishes(self) -> None:
        """第一轮模型请求 write_file 工具，第二轮模型给出文字总结 -> 验证工具真的被本地执行了。"""
        first = FakeMessage(
            content=[
                FakeBlock(type="text", text="我先创建文件。"),
                FakeBlock(
                    type="tool_use",
                    id="call_1",
                    name="write_file",
                    input={"path": "hello.txt", "content": "hello world"},
                ),
            ],
            stop_reason="tool_use",
        )
        second = FakeMessage(content=[FakeBlock(type="text", text="文件已创建完成。")], stop_reason="end_turn")

        agent = CodingAgent(config=self.config)
        with patch.object(agent.llm, "create_message", side_effect=[first, second]) as mocked:
            result = agent.run_task("创建一个 hello.txt")

        self.assertEqual(mocked.call_count, 2)
        self.assertTrue((self.tmp_dir / "hello.txt").exists())
        self.assertEqual((self.tmp_dir / "hello.txt").read_text(), "hello world")
        self.assertIn("已创建完成", result)

        # 校验对话历史里确实包含了 tool_result，且内容与工具真实返回一致
        messages = agent.conversation.get_messages()
        tool_result_msgs = [
            b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        self.assertEqual(len(tool_result_msgs), 1)
        self.assertFalse(tool_result_msgs[0]["is_error"])

    def test_max_iterations_terminates_infinite_tool_loop(self) -> None:
        """模型每轮都请求工具、永不给出最终文字 -> 必须在 max_iterations 后被强制终止，而不是死循环。"""
        always_tool_call = FakeMessage(
            content=[
                FakeBlock(
                    type="tool_use", id="call_x", name="list_dir", input={"path": "."}
                )
            ],
            stop_reason="tool_use",
        )
        agent = CodingAgent(config=self.config)
        with patch.object(agent.llm, "create_message", return_value=always_tool_call) as mocked:
            result = agent.run_task("一直循环下去")

        self.assertEqual(mocked.call_count, self.config.max_iterations)
        self.assertIn("已达到最大迭代轮数", result)


if __name__ == "__main__":
    unittest.main()
