#!/usr/bin/env python3
"""
命令行入口。

两种用法：
  1. 单任务模式：  python run.py -t "在 workspace 目录下写一个计算斐波那契数列的 fib.py"
  2. 交互模式：    python run.py
     进入后可以连续输入多个任务，每个任务独立跑完一轮"思考-行动"循环再等待下一次输入，
     输入 exit / quit 退出。
"""
from __future__ import annotations

import argparse
import sys

from agent.config import CONFIG
from agent.core import CodingAgent
from agent.logger import log_assistant_text, log_error, log_system, log_user


def main() -> None:
    parser = argparse.ArgumentParser(description="一个不依赖任何 Agent 框架的最小可用编程智能体")
    parser.add_argument("-t", "--task", type=str, default=None, help="直接指定一个任务并单次运行后退出")
    args = parser.parse_args()

    try:
        agent = CodingAgent(config=CONFIG)
    except RuntimeError as e:
        log_error(str(e))
        sys.exit(1)

    log_system(f"工作目录: {CONFIG.workspace_dir}")
    log_system(f"使用模型: {CONFIG.model}")

    if args.task:
        log_user(args.task)
        agent.run_task(args.task)
        # log_assistant_text(result)
        return

    log_system("已进入交互模式，输入任务描述后回车执行；输入 exit / quit 退出。")
    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            log_system("收到退出信号，再见。")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            log_system("再见。")
            break

        try:
            agent.run_task(user_input)
            # log_assistant_text(result)
        except KeyboardInterrupt:
            log_system("已中断当前任务（未退出程序），可以继续输入下一个任务。")


if __name__ == "__main__":
    main()
