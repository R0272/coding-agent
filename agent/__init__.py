"""
Mini Coding Agent

一个不依赖任何 Agent 框架 / Agent SDK 的最小可用编程智能体。

子模块：

- config.py
    读取环境变量并产生统一配置；

- llm_client.py
    使用 httpx 直接请求 OpenRouter REST API，
    并负责模型协议与项目内部格式之间的转换；

- context.py
    自行维护多轮对话历史与上下文压缩；

- core.py
    Agent 主循环，负责解析模型输出、
    调度本地工具和判断循环终止；

- tools/
    所有文件和命令工具的本地 Python 实现。

项目不依赖服务端托管的代码执行或文件工具。
"""

__version__ = "1.0.0"
