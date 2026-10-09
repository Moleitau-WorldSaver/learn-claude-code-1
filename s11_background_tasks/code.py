#!/usr/bin/env python3
"""
s11_background_tasks.py - Background Tasks

    Main thread                              Background thread
    +------------------------------+         +----------------------+
    | bash(run_in_background=True) | ------> | run command          |
    | return bg_id                 |         | queue result         |
    | continue agent loop          | <------ +----------------------+
    | next turn: collect           |
    +------------------------------+
"""

import atexit
import glob
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

ENVIRONMENT_PROMPT = (
    "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix "
    "Bash or PowerShell syntax, and prefer dedicated file tools for file operations"
    if os.name == "nt"
    else "Unix-like: the bash tool runs the system shell"
)
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use tools to solve tasks. "
    "Set run_in_background to true only for independent Bash commands."
)


# -- From s04: tool implementations --

# 等价于 _shell_processes = set()
# 初始化一个集合, 里面装Popen对象, 变量名字是_shell_processes
_shell_processes: set[subprocess.Popen] = set() 
# 创造一个RLock 锁的对象, 对象名是_shell_process_lock
_shell_process_lock = threading.RLock() 
 
# 停止子进程(停止harness起的进程)
def _stop_process_group(process: subprocess.Popen):
    """Stop a shell process and its children (cross-platform).

    POSIX uses process-group signals (SIGTERM then SIGKILL). Windows has
    neither ``os.killpg`` nor ``signal.SIGKILL``, so it falls back to
    ``Popen.terminate()`` / ``Popen.kill()``.
    """
    if os.name == "nt":
        # # 目标：把一个 shell 进程【连同它启动的子孙】尽量收掉
        # # 手段：先礼后兵（terminate/SIGTERM → kill/SIGKILL），每步都先确认 + 吞掉异常
        for stop in (process.terminate, process.kill): 
            if process.poll() is not None: # # 已经死了就别碰（幂等）
                return
            try:
                stop()
            except OSError: # # 进程没了/没权限 → 认了, 为什么认? 干活失败 → 要响亮, 收尾失败 → 要安静
                return 
            try:
                process.wait(timeout=0.05) # # 给 0.05 秒体面地死
            except subprocess.TimeoutExpired:
                continue #  # 没死？换更强的一招
        return

    # SIGKILL is POSIX-only; fall back to SIGTERM on platforms without it.
    for sig in (signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, OSError):
            return
        time.sleep(0.05)

# 停止所有子进程
def _stop_all_shell_processes():
    with _shell_process_lock:
        # 锁里抄一份完整名单：抄的时候不许别人动这个 set
        processes = list(_shell_processes)
    for process in processes:
        _stop_process_group(process)

# 以后我收到 SIGTERM，别用默认的'立刻死'，改成调用我写的这个函数
def _handle_termination_signal(signum, _frame):
    _stop_all_shell_processes()
    # 我收拾完了，现在退出；退出码写成 143，告诉外面我是被 15 号信号请走的。
    raise SystemExit(128 + signum) # 128 + N 是 Unix 惯例：被第 N 号信号杀死 → 退出码写 128 + N

#登记一下: 程序退出执行这个函数
atexit.register(_stop_all_shell_processes)
# 登记一下：以后收到 SIGTERM，就调用这个函数。
signal.signal(signal.SIGTERM, _handle_termination_signal)

# # 同步跑一条命令：起子进程 → 登记 → 等它（最多 120 秒）→ 返回 (它说了什么, 它是怎么结束的)
#! 子进程只跑命令, 模型并没有进入子进程
def _run_bash_process(command: str) -> tuple[str, int | None]:
    process = None
    try:
        # start_new_session (setsid) exists only on POSIX; skip it on Windows.
        # 按平台准备一份额外的启动参数：非 Windows 就要求子进程"自立门户"（新会话／新进程组），Windows 则什么都不加。
        popen_kwargs = {} if os.name == "nt" else {"start_new_session": True}
        # 真正启动那个子进程：交给系统 shell 执行这条命令、在指定目录跑、把它的输出接进管道，
        # 然后立刻返回一个代表它的 Popen 对象（不等它结束）
        # 一句话: 创造子进程, 返回Popen对象管理
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=WORKDIR,
            stdout=subprocess.PIPE,# 两行, 输出管子
            stderr=subprocess.PIPE,
            text=True, errors="replace",
            **popen_kwargs,
        )
        with _shell_process_lock:
            _shell_processes.add(process) # 填进进程set中, 里面存了好多Popen对象

        # 数据来的过程是流式的；但 output 是一次性拿到整块的。
        # 也就是说只要子进程在运行, 我们就一直卡在communicate这里, 流式拿信息
        stdout, stderr = process.communicate(timeout=120) # 把输出读干净, 最多等120s
        output = (stdout + stderr).strip() # 去掉首尾的空白

        # # 输出超 5 万字符就截断、空白就用占位；接上退出码，一起打包成元组返回
        return (output[:50000] if output else "(no output)"), process.returncode
    except subprocess.TimeoutExpired: # 超时
        return "Error: Timeout (120s)", None
    except OSError as error: # 子进程根本没起来
        return f"Error: {type(error).__name__}: {error}", None
    finally:# 无论如何都要走的收尾
        if process is not None:
            _stop_process_group(process) # 直接停
            try:
                process.wait(timeout=0.2) # 确认
            except subprocess.TimeoutExpired:
                pass
            with _shell_process_lock:
                _shell_processes.discard(process) # 在set中划掉

# return 0代表正常, 正常输出
def _format_bash_result(output: str, exit_code: int | None) -> str:
    if exit_code in (0, None):
        return output
    # 不正常, 错误行 + 原来输出
    return f"Error: command exited with status {exit_code}\n{output}"


def run_bash(command: str, run_in_background: bool = False) -> str:
    return _format_bash_result(*_run_bash_process(command))


def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as error:
        return f"Error: {error}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as error:
        return f"Error: {error}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as error:
        return f"Error: {error}"


def run_glob(pattern: str) -> str:
    try:
        matches = sorted({
            match
            for match in glob.glob(pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as error:
        return f"Error: {error}"


TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {
                          "command": {"type": "string"},
                          "run_in_background": {"type": "boolean"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# -- From s04: hooks and permission checks --

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def permission_hook(block):
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if contains_destructive_command(command) or any(
            keyword in command for keyword in DESTRUCTIVE
        ):
            print("\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"

    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print("\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None


def log_hook(block):
    preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({preview})\033[0m")
    return None


def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(
            f"\033[33m[HOOK] Large output from {block.name}: "
            f"{len(str(output))} chars\033[0m"
        )
    return None


def context_inject_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list):
    tool_count = sum(
        1
        for message in messages
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def call_tool(block) -> str:
    handler = TOOL_HANDLERS.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as error:
        output = f"Error: {error}"
    return str(output)


# -- New in s11: background execution --
# 只管理后台任务的类
class BackgroundManager:
    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self.results: dict[str, str] = {}
        self._ready: list[str] = []
        self._counter = 0 # 发号器
        self._lock = threading.Lock()

    # 用block块来起线程, 跑子进程任务, 返回的是任务id
    def start(self, block) -> str:
        if block.name != "bash": # 只允许bash
            raise ValueError("Only Bash commands can run in the background")
        command = block.input.get("command")
        if not isinstance(command, str) or not command.strip(): # 检查类型对不对 + 内容是不是空的
            raise ValueError("Bash command cannot be empty")
        
        # 后台要同时跑多条命令 -> 每条命令一条线程, 之后这个线程会执行创建Popen,然后起进程
        # 于是这本账会被两种线程碰：主线程负责发号/登记/取走，后台线程负责回写状态和结果
        # 碰的是同一份数据（tasks / _ready / _counter）-> 用锁
        with self._lock: 
            self._counter += 1 # 发号
            # 登记
            task_id = f"bg_{self._counter:04d}" 
            self.tasks[task_id] = { 
                "tool_use_id": block.id,
                "command": command,
                "status": "running",
            }
        # 起一个线程, 用来创建子进程任务
        thread = threading.Thread(
            target=self._run, # 线程跑起来立马调用这个函数
            args=(task_id, command), # 传给那个函数的参数(必须写成元组)
            daemon=True, # 守护线程标志(也就是说主程序退出了, 直接掐灭这个,程序不等)
        )
        try:
            thread.start()
        except Exception:
            with self._lock:
                self.tasks.pop(task_id, None)
            raise
        print(f"  [background] started {task_id}: {command[:60]}")
        return task_id

    def _run(self, task_id: str, command: str):
        try:
            # 跑bash命令, 拿结果
            output, exit_code = _run_bash_process(command)
            result = _format_bash_result(output, exit_code)
            # 如果是返回0的状态跑完, 任务就是顺利完成
            status = "completed" if exit_code == 0 else "failed"
        except Exception as error:
            result = f"Error: {type(error).__name__}: {error}"
            status = "failed"

        # 将结果传送出去
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return
            task["status"] = status
            self.results[task_id] = result
            self._ready.append(task_id) # _ready里放待通知队列(放着已经跑完, 还没通知模型的任务编号)

    def collect(self) -> list[str]:
        with self._lock: # 从锁里取件
            ready = []
            for task_id in self._ready: # # 遍历"可取件"清单
                task = self.tasks.pop(task_id, None) # 从登记本上摘下来（连记录一起删）
                result = self.results.pop(task_id, "") # # 把结果也取走
                if task is not None:
                    ready.append((task_id, task, result)) # 存着大量信息
            self._ready.clear() # 清空

        notifications = []
        for task_id, task, result in ready: # 将信息塞到文本中, 返回
            notifications.append(
                f"<task_notification>\n"
                f"  <task_id>{task_id}</task_id>\n"
                f"  <status>{task['status']}</status>\n"
                f"  <command>{task['command']}</command>\n"
                f"  <summary>{result[:500]}</summary>\n"
                f"</task_notification>"
            )
            print(f"  [background] collected {task_id}: {task['status']}")
        return notifications


BACKGROUND = BackgroundManager()
background_tasks = BACKGROUND.tasks
background_results = BACKGROUND.results

# 拿工具调用块的name, 和 input的参数
def should_run_background(tool_name: str, tool_input: dict) -> bool:
    return (
        tool_name == "bash" # ① 必须是 bash 工具
        and tool_input.get("run_in_background") is True # ② 取出run_in_background这一项 且 这一项是True
    )


def start_background_task(block) -> str:
    return BACKGROUND.start(block)


def collect_background_results() -> list[str]:
    return BACKGROUND.collect()

# 当通过后台管理任务的类，里的具体收集函数拿到的文本，塞到模型里。
def inject_background_results(messages: list) -> int:
    notifications = collect_background_results()
    if not notifications:
        return 0

    blocks = [{"type": "text", "text": item} for item in notifications]
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            content.extend(blocks)
        else:
            messages[-1]["content"] = [
                {"type": "text", "text": str(content)},
                *blocks,
            ]
    else:
        messages.append({"role": "user", "content": blocks})
    return len(notifications)


def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked is not None:
        return str(blocked)

    if should_run_background(block.name, block.input):
        try:
            task_id = start_background_task(block)
            output = (
                f"[Background task {task_id} started] "
                "The result will be collected on a later turn."
            )
        except Exception as error:
            output = f"Error: {error}"
    else:
        output = call_tool(block)

    trigger_hooks("PostToolUse", block, output)
    return output


# -- Agent loop --

def agent_loop(messages: list):
    while True:
        inject_background_results(messages) # 塞模型里对话
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        results = []
        for block in tool_calls:
            output = execute_tool(block)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s11: Background Tasks - explicit background Bash execution")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s11 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
