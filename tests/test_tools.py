"""
工具层的单元测试。

这些测试完全不需要调用大模型 API（不需要联网、不需要 API Key），
只验证本项目自己实现的"本地工具执行"与"路径沙箱"逻辑是否正确。

运行方式：
    python -m unittest tests/test_tools.py -v
或（若已安装 pytest）：
    pytest tests/test_tools.py -v
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
import sys

from pathlib import Path

from agent.config import AgentConfig
from agent.tools import build_default_registry
from agent.tools.sandbox import PathEscapeError, resolve_safe_path


class ToolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="agent_test_"))
        self.config = AgentConfig(
            api_key="dummy-not-used-in-these-tests",
            workspace_dir=self.tmp_dir,
        )
        self.registry = build_default_registry()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def test_write_then_read_file(self) -> None:
        result = self.registry.execute("write_file", {"path": "hello.py", "content": "print('hi')\n"}, self.config)
        self.assertFalse(result.is_error)
        self.assertTrue((self.tmp_dir / "hello.py").exists())

        read_result = self.registry.execute("read_file", {"path": "hello.py"}, self.config)
        self.assertFalse(read_result.is_error)
        self.assertIn("print('hi')", read_result.output)
        self.assertIn("1\t", read_result.output)  # 带行号

    def test_read_nonexistent_file_returns_error_not_exception(self) -> None:
        result = self.registry.execute("read_file", {"path": "not_exist.py"}, self.config)
        self.assertTrue(result.is_error)
        self.assertIn("不存在", result.output)

    def test_edit_file_unique_match(self) -> None:
        self.registry.execute("write_file", {"path": "a.txt", "content": "foo\nbar\nfoo\n"}, self.config)
        # "foo" 出现两次，不唯一，应当拒绝
        ambiguous = self.registry.execute(
            "edit_file", {"path": "a.txt", "old_str": "foo", "new_str": "baz"}, self.config
        )
        self.assertTrue(ambiguous.is_error)
        self.assertIn("出现了 2 次", ambiguous.output)

        # 提供带上下文、唯一匹配的 old_str 才能成功
        ok = self.registry.execute(
            "edit_file", {"path": "a.txt", "old_str": "bar\nfoo", "new_str": "bar\nbaz"}, self.config
        )
        self.assertFalse(ok.is_error)
        self.assertEqual((self.tmp_dir / "a.txt").read_text(), "foo\nbar\nbaz\n")

    def test_path_escape_is_blocked(self) -> None:
        with self.assertRaises(PathEscapeError):
            resolve_safe_path("../../etc/passwd", self.tmp_dir)

        result = self.registry.execute("read_file", {"path": "../outside.txt"}, self.config)
        self.assertTrue(result.is_error)
        self.assertIn("超出了工作目录", result.output)

    def test_list_dir(self) -> None:
        (self.tmp_dir / "sub").mkdir()
        (self.tmp_dir / "sub" / "x.py").write_text("x = 1\n")
        result = self.registry.execute("list_dir", {"path": "."}, self.config)
        self.assertFalse(result.is_error)
        self.assertIn("sub/", result.output)
        self.assertIn("x.py", result.output)

    def test_run_bash_basic(self) -> None:
        result = self.registry.execute("run_bash", {"command": "echo hello_agent"}, self.config)
        self.assertFalse(result.is_error)
        self.assertIn("hello_agent", result.output)
        self.assertIn("退出码: 0", result.output)

    def test_run_bash_timeout(self) -> None:
        command = f'"{sys.executable}" -c "import time; time.sleep(5)"'
        result = self.registry.execute(
            "run_bash",
            {"command": command, "timeout": 1},
            self.config,
        )
        self.assertTrue(result.is_error)
        self.assertIn("超过 1 秒", result.output)


    def test_unknown_tool_returns_error_not_exception(self) -> None:
        result = self.registry.execute("does_not_exist", {}, self.config)
        self.assertTrue(result.is_error)
        self.assertIn("未知工具", result.output)


if __name__ == "__main__":
    unittest.main()
