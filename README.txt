Coding Agent

Git 仓库：
https://github.com/R0272/coding-agent

一、项目简介

本项目实现了一个不依赖任何 Agent 框架的最小可用编程智能体。程序通过大语言模型的原生 tool calling 能力进行决策，由本地 Python 代码真正执行工具，从而完成读写文件、精确修改代码、查看目录和执行命令等编程任务。

模型请求通过 OpenRouter API 完成，默认使用支持 tool calling 的 Qwen3-Coder 免费模型。项目中的 Agent 主循环、对话历史与上下文压缩、工具定义与执行、模型输出解析、循环终止条件及错误处理均自行实现，未使用 LangChain、AutoGen、Claude Agent SDK 等 Agent 框架，也未使用服务端托管的代码执行或文件工具。

二、环境

Python 3.10+

安装依赖：

pip install -r requirements.txt

在项目根目录创建 .env：

OPENROUTER_API_KEY=你的_API_Key

API Key 不提交至代码仓库。

三、运行

单任务模式：

python run.py -t "在工作目录下创建 fib.py，实现斐波那契数列并编写测试验证"

交互模式：

python run.py

运行过程中可观察模型请求工具、本地执行工具、工具结果返回模型以及模型继续决策的完整循环。模型创建和修改的文件限制在 workspace/ 沙箱目录中。

四、测试

python -m unittest discover tests -v

测试覆盖本地工具、路径越权防护、命令超时、上下文压缩及 Agent 主循环终止条件等逻辑。

五、特色

1. 五个本地工具全部自行实现；
2. workspace 沙箱限制文件访问范围；
3. edit_file 仅在目标文本唯一匹配时执行；
4. 自行维护多轮 tool_use / tool_result 历史；
5. 上下文过长时压缩旧工具输出；
6. 网络临时错误采用指数退避重试；
7. 最大循环轮数防止 Agent 无限执行。
