#!/usr/bin/env python3
"""
s10_task_system.py - Task System

    .tasks/
      task_a1b2c3d4.json  {status: completed, blockedBy: []}
      task_e5f6a7b8.json  {status: pending, blockedBy: [task_a1b2c3d4]}
      task_11223344.json  {status: pending, blockedBy: [task_e5f6a7b8]}

    Dependency graph:

    +-----------+      +-----------+      +-----------+
    | schema    | ---> | API       | ---> | tests     |
    | completed |      | pending   |      | pending   |
    +-----------+      +-----------+      +-----------+

    can_start(API) is true because schema is completed.

    Task lifecycle:

    pending --claim_task--> in_progress --complete_task--> completed
"""

import glob
import json
import os
import re
import secrets
import subprocess
from dataclasses import asdict, dataclass
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
    "Use task tools to track dependencies and progress. Create all task nodes "
    "first. After create_task returns runtime-generated IDs, use update_task "
    "with those exact IDs to add dependencies."
)


# -- New in s10: persistent task records --

TASKS_DIR = WORKDIR / ".tasks"
# 任务 ID 格式必须是 Task_ + 8 位十六进制
TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{8}$")


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]


class TaskStore:
    def __init__(self, directory: Path):
        self.directory = directory

    # create=True 时先建目录(已存在也不报错),注意这一步在检查之前
    # 把目录展开成绝对路径
    # 展开后必须落在 WORKDIR 里面,否则报错
    # 通过则返回
    def _root(self, create: bool = False) -> Path:
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        root = self.directory.resolve()
        if not root.is_relative_to(WORKDIR.resolve()):
            raise ValueError("Task store escapes the workspace")
        return root

    def _path(self, task_id: str, create_root: bool = False) -> Path:
        # 检查是否是字符串, 或者格式是否匹配
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError(f"Invalid task ID: {task_id!r}")
        root = self._root(create=create_root)
        path = (root / f"{task_id}.json").resolve()# 绝对路径地址
        if not path.is_relative_to(root):# 如果没在root里, 报错
            raise ValueError(f"Invalid task ID: {task_id!r}")
        return path

    # 看文件在不在
    def exists(self, task_id: str) -> bool:
        return self._path(task_id).is_file()

    # 创建一个任务文件
    def create(self, subject: str, description: str = "") -> Task:
        subject = subject.strip() # 提出subject, 也就是任务的主题
        if not subject:# 为空, 报错
            raise ValueError("Task subject cannot be empty")

        self._root(create=True) # 创建根目录
        for _ in range(100): #最多循环100次
            task = Task(
                id=f"task_{secrets.token_hex(4)}", # 造一个 8 位的、不可预测的、纯十六进制的短字符串
                subject=subject,
                description=description,
                status="pending",
                owner=None,
                blockedBy=[],
            )
            try:
                # with管理文件对象
                with self._path(task.id, create_root=True).open( # open("x")是排他创建, 防止了任务明重复
                    "x", encoding="utf-8"
                ) as handle:
                    # asdict(task): 按字段表取出该实例当前的值 → 新的 dict（键来自字段表，值来自实例）
                    # json.dump(d, handle): 把 dict 编码成 JSON 并直接写进 handle —— 注意是 dump 不是 dumps，
                    #                       它不返回字符串，而是边编码边往文件对象写
                    # indent=2: 带缩进的可读格式，否则是一整行
                    json.dump(asdict(task), handle, indent=2)
                return task
            except FileExistsError: 
                continue # 如果已经存在, 换一个
        raise RuntimeError("Could not allocate a unique task ID")

    # 检查 task是否依赖于target(也就是第一个任务是否依赖于第二个任务)
    def _depends_on(self, task_id: str, target_id: str) -> bool:
        """Return whether task_id transitively depends on target_id."""
        pending = [task_id] # 待查列表, 先放自己
        visited = set() # 防已经查过的
        while pending:
            current = pending.pop() # pop
            if current == target_id: # 找到了
                return True # 返回true
            if current in visited: #如果检查过, 跳过 
                continue 
            visited.add(current) # 加上查过的
            pending.extend(self.load(current).blockedBy) # 加上依赖的所有节点
        return False

    def update_dependencies(self, task_id: str,
                            add_blocked_by: list[str]) -> Task:
        if not isinstance(add_blocked_by, list): #必须传入列表
            raise ValueError("addBlockedBy must be a list of task IDs")

        task = self.load(task_id)
        if task.status != "pending" or task.owner is not None: # 任务必须是pendin且无人认领
            raise ValueError(
                f"Task {task_id} dependencies can only be updated while "
                "pending and unowned"
            )
        # 借字典的键去重：元素当键（故必须可哈希），字典保序所以顺序不变，最后取出键拼成列表
        dependencies = list(dict.fromkeys(add_blocked_by))
        for dependency in dependencies: # 遍历任务名的列表
            if dependency == task_id: # 这个是依赖列表, 自己不能依赖自己
                raise ValueError("Task cannot depend on itself")
            if not self.exists(dependency): # 依赖的具体任务必须存在
                raise ValueError(f"Dependency not found: {dependency}")
            # 新边 + 要加的依赖边 依赖我 = 报错
            if dependency not in task.blockedBy and self._depends_on( 
                dependency, task_id # 确保我当前加的依赖, 并没有依赖我, 如果依赖了我报错
            ):
                raise ValueError(
                    f"Dependency cycle detected: {task_id} -> {dependency}"
                )

        task.blockedBy.extend(
            dependency for dependency in dependencies
            if dependency not in task.blockedBy # 只加还没有的依赖, 把加相同依赖的边的变成无害操作
        )
        self.save(task)
        return task

    # 把任务写入文件
    def save(self, task: Task) -> None:
        # 将任务信息写到任务id对应的文件里
        self._path(task.id, create_root=True).write_text(
            json.dumps(asdict(task), indent=2), # 这里会编成json，返回字符串
            encoding="utf-8",
        )
    # 加载是通过id->文件地址, 然后读文件内容-> json
    def load(self, task_id: str) -> Task:
        data = json.loads(self._path(task_id).read_text(encoding="utf-8"))
        task = Task(**data) # 把 dict 摊开成关键字实参, 这里等效: Task(id="task_a1b2c3d4", subject="schema", description="建表",
                                                            ## status="pending", owner=None, blockedBy=[])
        if task.id != task_id: # 如果创造的不对(我们是根据信息创造的, 正常不可能不相等)
            raise ValueError(f"Task file ID does not match {task_id}")
        if task.status not in ("pending", "in_progress", "completed"): # 不在三个状态中
            raise ValueError(f"Invalid task status: {task.status}")
        return task # 返回创造的任务对象

    # 遍历所有任务, 装到list[Task]中
    def list(self) -> list[Task]:
        if not self.directory.exists(): #文件夹不存在, 回空
            return []
        root = self._root()
        # 通过glob找到task_*.json, 然后排序遍历, 拿地址前的(也就是任务id), 依次加载到list中
        return [self.load(path.stem)
                for path in sorted(root.glob("task_*.json"))]


TASKS = TaskStore(TASKS_DIR)


def create_task(subject: str, description: str = "") -> Task:
    return TASKS.create(subject, description)


def update_task(task_id: str, addBlockedBy: list[str]) -> Task:
    return TASKS.update_dependencies(task_id, addBlockedBy)


def load_task(task_id: str) -> Task:
    return TASKS.load(task_id)


def list_tasks() -> list[Task]:
    return TASKS.list()

# 得到任务文本里的内容
def get_task(task_id: str) -> str:
    return json.dumps(asdict(load_task(task_id)), indent=2)

# 返回依赖中没有完成的任务
def incomplete_dependencies(task: Task) -> list[str]:
    incomplete = [] # 放的是哪些依赖没有完成的任务名
    for dependency in task.blockedBy:
        try:
            if load_task(dependency).status != "completed":
                incomplete.append(dependency) # 如果不是已完成, 加
        except (FileNotFoundError, ValueError):
            incomplete.append(dependency)
    return incomplete

# 依赖全部完成 -> True -> 可以开工
def can_start(task_id: str) -> bool:
    return not incomplete_dependencies(load_task(task_id))

# 认领任务
def claim_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "pending": # 是否是未处理的, 不是未处理的不能领
        return f"Task {task_id} is {task.status}, cannot claim"
    dependencies = incomplete_dependencies(task) # 拿到没有完成的任务
    if dependencies: # 如果有没有完成的任务
        return f"Blocked by: {dependencies}" # 也直接返回
    task.owner = owner # 认领
    task.status = "in_progress" # 改状态
    TASKS.save(task) # 保存任务
    print(f"  [claim] {task.subject} -> in_progress (owner: {owner})") # 打印字段
    return f"Claimed {task.id} ({task.subject})" # 返回字段

# 完成任务, 打印和返回消息
def complete_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "in_progress": # 是否进行
        return f"Task {task_id} is {task.status}, cannot complete"
    if task.owner != owner: # 是否是任务的持有者
        return f"Task {task_id} is owned by {task.owner}, not {owner}"
    ready_before = {  # 完成前可以开工的(可认领的)
        candidate.id
        for candidate in list_tasks() # 遍历整个任务列表
        if candidate.status == "pending" # 如果是未处理
        and candidate.blockedBy # 当前遍历的有依赖
        and can_start(candidate.id) # 可以开工
    }
    task.status = "completed" # == 更改状态, 有的解锁的
    TASKS.save(task) # 真正改磁盘里的Task状态
    
    # 当前任务完成后, 更改状态后可认领的名单会变多, 我们拿取差值
    unblocked = [candidate.subject for candidate in list_tasks()
                 if candidate.status == "pending"
                 and candidate.blockedBy
                 and candidate.id not in ready_before # 差集 = 本次解锁的
                 and can_start(candidate.id)]
    print(f"  [complete] {task.subject}")
    message = f"Completed {task.id} ({task.subject})"
    if unblocked:
        message += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  [unblocked] {', '.join(unblocked)}")
    return message


# -- From s04: tool implementations --

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True, errors="replace",
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
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


def run_create_task(subject: str, description: str = "") -> str:
    task = create_task(subject, description)
    print(f"  [create] {task.subject}")
    return f"Created {task.id}: {task.subject}"


def run_update_task(task_id: str, addBlockedBy: list[str]) -> str:
    task = update_task(task_id, addBlockedBy)
    dependencies = ", ".join(task.blockedBy) or "(none)"
    print(f"  [update] {task.subject} blockedBy: {dependencies}")
    return f"Updated {task.id} blockedBy: {dependencies}"


def run_list_tasks() -> str:
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for task in tasks:
        marker = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }.get(task.status, "[?]")
        dependencies = (
            f" (blockedBy: {', '.join(task.blockedBy)})"
            if task.blockedBy else ""
        )
        owner = f" [{task.owner}]" if task.owner else ""
        lines.append(
            f"{marker} {task.id}: {task.subject} "
            f"[{task.status}]{owner}{dependencies}"
        )
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    return get_task(task_id)


def run_claim_task(task_id: str) -> str:
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    return complete_task(task_id, owner="agent")


TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "create_task", "description": "Create a task and return its runtime-generated ID.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"], "additionalProperties": False}},
    {"name": "update_task", "description": "Add dependencies using IDs returned by create_task.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string", "pattern": "^task_[0-9a-f]{8}$"}, "addBlockedBy": {"type": "array", "items": {"type": "string", "pattern": "^task_[0-9a-f]{8}$"}, "minItems": 1}}, "required": ["task_id", "addBlockedBy"], "additionalProperties": False}},
    {"name": "list_tasks", "description": "List tasks with status, owner, and dependencies.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_task", "description": "Get a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "claim_task", "description": "Claim a pending task whose dependencies are complete.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "complete_task", "description": "Complete the task claimed by this agent.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "create_task": run_create_task,
    "update_task": run_update_task,
    "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task,
    "complete_task": run_complete_task,
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


def context_hook(query: str):
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


register_hook("UserPromptSubmit", context_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked)

    handler = TOOL_HANDLERS.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as error:
        output = f"Error: {error}"

    trigger_hooks("PostToolUse", block, output)
    return str(output)


# -- Agent loop --

def agent_loop(messages: list):
    while True:
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
    print("s10: Task System - dependencies and task state")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s10 >> \001\033[0m\002")
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
