#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while True:
        response = LLM(messages, tools)
        if response contains no tool_use:
            break
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Later chapters add policy,
hooks, and lifecycle controls around it.

Usage:
    pip install anthropic python-dotenv
    ANTHROPIC_API_KEY=... python s01_agent_loop/code.py
"""

import os
import subprocess

try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
# client 是Anthropic 类的实例, 是一个对象, 可以调用这个对象的一些函数, 把我的请求变成http发出去, 同时也可以接受服务器传过来的json
# 如果想要得到message, 就调用 client.messages.create() 这个函数, 传入参数, 就可以得到message
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

ENVIRONMENT_PROMPT = (
    "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix "
    "Bash or PowerShell syntax"
    if os.name == "nt"
    else "Unix-like: the bash tool runs the system shell"
)
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use bash to solve tasks. Act, don't explain."
)

# -- Tool definition: just bash --
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


# -- Tool execution --
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    # 主循环
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # Append assistant turn
        # 将模型输出的再次添加到消息列表中, 以便下一轮循环使用
        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't call a tool, we're done
        # 从response中遍历所有的块, 如果是工具调用类型, 就把它们放到tool_calls列表中
        tool_calls = [
            # 这里的语法: [ 放进去的东西   for 循环变量 in 来源列表   if 过滤条件 ]
            block for block in response.content if block.type == "tool_use"
        ]
        # 如果没有调用工具, 就结束循环
        if not tool_calls:
            return

        # Execute each tool call, collect results
        # 执行每个工具调用, 收集结果, 用results列表存储每个工具调用的结果, 
        # 每个结果是一个字典, 包含返回类型, 工具调用的id和输出内容
        results = []
        for block in tool_calls:
        # block.name    # 工具名,如 "bash"     ← 字符串
        # block.id      # 这次调用的唯一编号     ← 字符串,如 "toolu_01..."
        # block.input   # 模型为这个工具准备的参数 ← 字典! # block.input == {"command": "ls"}

            print(f"\033[33m$ {block.input['command']}\033[0m")
            output = run_bash(block.input["command"])
            print(output[:200])
            results.append({
                "type": "tool_result",
                # 为什么得有tool_use_id? 
                # 协议强制, 为什么强制, 其实是三方面:
                # 1, 并发冲突: 如果有多个工具且是并发的, 顺序插入可能会错乱, 需要明确标识哪个工具调用的结果
                # 2, 跳过缺位: 如果上限次数没结果, 工具将会跳过, 需要明确标识哪个工具调用的结果
                # 3, 跨轮次: 当前轮的 tool_use 不一定在当前轮就有 tool_result, id就是多轮次的联系
                "tool_use_id": block.id,
                "content": output,
            })

        # Feed tool results back, loop continues
        # 将工具调用的结果再次添加到消息列表中, 以便下一轮循环使用
        messages.append({"role": "user", "content": results})


# -- Entry point --
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s01 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response
        # -1 就是历史最后一轮对话输出的结果, 我们把这个结果输出
        response_content = history[-1]["content"]
        # 遍历content, 如果是文本类型, 就输出文本内容
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
