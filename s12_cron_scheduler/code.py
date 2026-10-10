#!/usr/bin/env python3
"""
s12_cron_scheduler.py - Cron Scheduler

    +--------------------------+   09:00   +-----------------------+
    | 0 9 * * *               | --------> | [Scheduled] run tests |
    | prompt: "run tests"      |           +-----------+-----------+
    +--------------------------+                       |
          scheduled_jobs                    cron_queue | agent idle
                                                        v
                                                +-------------+
                                                | Agent Loop  |
                                                +-------------+
"""

import glob
import json
import os
import re
import secrets
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
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
DURABLE_PATH = WORKDIR / ".scheduled_tasks.json" # 储存定时任务的地址
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
    "Use schedule_cron for work that should start at a future local time."
)


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
        if result.returncode != 0:
            return f"Error: command exited with status {result.returncode}\n{output}"
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


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

# 工具列表
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
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
        result = callback(*args) # 调用它，参数原样转交，接住它的返回值
        if result is not None:
            return result # 如果不是空, 有话说
    return None # 没话说


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))

# 请求用户权限
def request_permission(block, reason: str) -> str | None:
    # 本课新增：不是主线程 -> 直接拒绝，连问都不问
    if threading.current_thread() is not threading.main_thread():
        return "Permission denied: scheduled turns cannot request interactive approval"

    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   Tool: {block.name}({block.input})")
    choice = input("   Allow? [y/N] ").strip().lower()
    if choice not in ("y", "yes"):
        return "Permission denied by user"
    return None

# 如果请求失败, 返回Permission denied by user
# 请求成功, 返回None, 也就是没话说
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
            return request_permission(block, "Potentially destructive command")

    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            return request_permission(block, "Access outside workspace")
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


# -- New in s12: cron jobs --

# 定时任务类
@dataclass
class CronJob:
    id: str
    cron: str # 具体实行时间(五段式)
    prompt: str
    recurring: bool # 重复性任务?
    durable: bool # 落盘？（True=durable，重启还在；False=session，进程退出即消失）

    pending_delivery: bool = False # true = 到点了、已经排队，但话还没送到模型手上, false 模型接受
    last_fired: str | None = None # 最后一次触发是在哪一分钟"——存成 "2026-10-10 10:21" 这种字符串


scheduled_jobs: dict[str, CronJob] = {}  # 任务表
cron_queue: list[CronJob] = [] # 投递队列
cron_lock = threading.RLock()  # 保护上面两个 + 那个磁盘文件(DURABLE_PATH)

# DURABLE_PATH
# durable=True 的任务落盘在这里；创建 / 入队 / 确认 / 取消 四个时刻各写一次
# 存的是任务定义 + 投递状态（没有执行结果）；durable=False 的任务从不写这里

def _cron_field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        return value % int(field[2:]) == 0
    if "," in field:
        return any(_cron_field_matches(part.strip(), value)
                   for part in field.split(","))
    if "-" in field:
        start, end = field.split("-", 1)
        return int(start) <= value <= int(end)
    return value == int(field)

# 现在这个时间,符不符合这条 cron 表达式?
def cron_matches(cron_expr: str, moment: datetime) -> bool:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False

    minute, hour, day, month, weekday = fields
    cron_weekday = (moment.weekday() + 1) % 7
    if not (
        _cron_field_matches(minute, moment.minute)
        and _cron_field_matches(hour, moment.hour)
        and _cron_field_matches(month, moment.month)
    ):
        return False

    day_matches = _cron_field_matches(day, moment.day)
    weekday_matches = _cron_field_matches(weekday, cron_weekday)
    if day == "*" and weekday == "*":
        return True
    if day == "*":
        return weekday_matches
    if weekday == "*":
        return day_matches
    return day_matches or weekday_matches


def _validate_cron_field(field: str, minimum: int, maximum: int) -> str | None:
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) <= 0:
            return f"Invalid step: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            error = _validate_cron_field(part.strip(), minimum, maximum)
            if error:
                return error
        return None
    if "-" in field:
        start, end = field.split("-", 1)
        if not start.isdigit() or not end.isdigit():
            return f"Invalid range: {field}"
        start_value, end_value = int(start), int(end)
        if start_value > end_value:
            return f"Range start is greater than end: {field}"
        if start_value < minimum or end_value > maximum:
            return f"Range {field} is outside [{minimum}-{maximum}]"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    value = int(field)
    if value < minimum or value > maximum:
        return f"Value {value} is outside [{minimum}-{maximum}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"

    field_rules = [
        ("minute", 0, 59),
        ("hour", 0, 23),
        ("day-of-month", 1, 31),
        ("month", 1, 12),
        ("day-of-week", 0, 6),
    ]
    for field, (name, minimum, maximum) in zip(fields, field_rules):
        error = _validate_cron_field(field, minimum, maximum)
        if error:
            return f"{name}: {error}"
    return None


def save_durable_jobs():
    with cron_lock:
        payload = [
            asdict(job)
            for job in scheduled_jobs.values()
            if job.durable
        ]
        temporary = DURABLE_PATH.with_name(
            f"{DURABLE_PATH.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, DURABLE_PATH)
        finally:
            temporary.unlink(missing_ok=True)

# 启动时把磁盘上的定时任务读回内存
def load_durable_jobs():
    if not DURABLE_PATH.exists():
        return
    try:
        payload = json.loads(DURABLE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("expected a JSON list")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"  [cron] could not load {DURABLE_PATH.name}: {error}")
        return

    loaded = 0
    with cron_lock:
        for item in payload:
            try:
                job = CronJob(**item)
                error = validate_cron(job.cron)
                if error:
                    raise ValueError(error)
                if not job.id.startswith("cron_"):
                    raise ValueError("invalid job ID")
                if not job.prompt.strip():
                    raise ValueError("prompt cannot be empty")
            except (TypeError, ValueError) as error:
                print(f"  [cron] skipped invalid saved job: {error}")
                continue
            scheduled_jobs[job.id] = job
            if job.pending_delivery:
                cron_queue.append(job)
            loaded += 1
    if loaded:
        print(f"  [cron] loaded {loaded} durable job(s)")


def new_cron_id() -> str:
    for _ in range(100):
        job_id = f"cron_{secrets.token_hex(4)}"
        if job_id not in scheduled_jobs:
            return job_id
    raise RuntimeError("Could not allocate a cron job ID")


def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    error = validate_cron(cron)
    if error:
        return error
    if not prompt.strip():
        return "Prompt cannot be empty"

    with cron_lock:
        job = CronJob(
            id=new_cron_id(),
            cron=cron,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
        )
        scheduled_jobs[job.id] = job
        try:
            if durable:
                save_durable_jobs()
        except Exception:
            scheduled_jobs.pop(job.id, None)
            raise
    print(f"  [cron] scheduled {job.id}: {cron} -> {prompt[:60]}")
    return job


def cancel_job(job_id: str) -> str:
    with cron_lock:
        job = scheduled_jobs.get(job_id)
        if job is None:
            return f"Job {job_id} not found"

        previous_queue = list(cron_queue)
        scheduled_jobs.pop(job_id)
        cron_queue[:] = [queued for queued in cron_queue if queued.id != job_id]
        try:
            if job.durable:
                save_durable_jobs()
        except Exception:
            scheduled_jobs[job_id] = job
            cron_queue[:] = previous_queue
            raise
    print(f"  [cron] cancelled {job_id}")
    return f"Cancelled {job_id}"

# 如果有到点了, 先落盘, 然后放入cron_queue中
def _enqueue_due_job(job: CronJob, minute_marker: str | None = None):
    old_pending = job.pending_delivery
    old_last_fired = job.last_fired
    job.pending_delivery = True
    if minute_marker is not None:
        job.last_fired = minute_marker
    try:
        if job.durable:
            save_durable_jobs()
    except Exception:
        job.pending_delivery = old_pending
        job.last_fired = old_last_fired
        raise
    cron_queue.append(job)

# 每秒检查, 但是每分钟只放一次
# 看一眼现在几点,把到点的定时任务扔进队列。
def poll_due_jobs(moment: datetime):
    minute_marker = moment.strftime("%Y-%m-%d %H:%M")
    with cron_lock:
        for job in list(scheduled_jobs.values()): 
            try:
                if job.pending_delivery or job.last_fired == minute_marker:
                    continue # 这个任务已经排过队了/这一分钟已经触发过了 → 跳过
                if cron_matches(job.cron, moment): # 
                    _enqueue_due_job(job, minute_marker) # # 到点了 → 放进队列
                    print(f"  [cron] due {job.id}: {job.prompt[:60]}")
            except Exception as error:
                print(f"  [cron] could not enqueue {job.id}: {error}")

# 创造一个消费队列
def consume_cron_queue() -> list[CronJob]:
    with cron_lock: # 锁保护
        jobs = list(cron_queue) # 从投递队列中拿信息
        cron_queue.clear()
    return jobs

# 确认这批任务模型已经收到, 重复任务清掉pending，一次性任务删掉, 然后把结果落到磁盘, 磁盘写失败就整体回滚。
def acknowledge_cron_jobs(jobs: list[CronJob]):
    changed: list[tuple[CronJob, bool]] = [] # 记录本A：存元组, 改过 pending 的任务 + 改之前的旧值(通用习惯)
    removed: list[CronJob] = [] # 记录本B：被删掉的任务
    with cron_lock:
        for delivered in jobs:
            current = scheduled_jobs.get(delivered.id) # 拿具体任务
            if current is None: # 表里没有, 跳过
                continue
            changed.append((current, current.pending_delivery)) # 备份旧数值
            if current.recurring: # 如果是重复性任务
                current.pending_delivery = False # 标成模型接受
            else:
                removed.append(current) # 加入删除里, 回滚用
                scheduled_jobs.pop(current.id) # pop出

        try:
            if any(job.durable for job, _ in changed): # 统一落盘
                save_durable_jobs()
        except Exception: #回滚
            for job in removed:
                scheduled_jobs[job.id] = job #    把删掉的塞回表
            for job, pending in changed: #    把 pending 恢复旧值
                job.pending_delivery = pending
            queued_ids = {job.id for job in cron_queue} #    准备重新入队
            for job, _ in changed:
                if job.id not in queued_ids: #    还不在队列里才加
                    cron_queue.append(job)
            raise

# 将任务标回 "未送达" + 重新排队
def restore_cron_jobs(jobs: list[CronJob]):
    """交付失败：把这批任务退回队列（只动内存，盘上本来就是未送达）"""
    with cron_lock:
        queued_ids = {job.id for job in cron_queue} # 当前排好队的那些id, 用{}(也就是set存)
        for delivered in jobs: # 遍历
            current = scheduled_jobs.get(delivered.id) # 看任务表, 拿id对应的任务
            if current is None:
                continue # 空, 跳过
            current.pending_delivery = True # true = 到点了、已经排队，但话还没送到模型手上, false才是送到了, 这里标成没送到
            if current.id not in queued_ids: # 如果没在队列中
                cron_queue.append(current) # 加上任务
                queued_ids.add(current.id) # 排好序的set也加上号


def has_cron_queue() -> bool: # 检查有无东西
    with cron_lock:
        return bool(cron_queue)


def run_schedule_cron(cron: str, prompt: str, recurring: bool = True,
                      durable: bool = True) -> str:
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: {cron} -> {prompt}"


def run_list_crons() -> str:
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs."

    lines = []
    for job in jobs:
        frequency = "recurring" if job.recurring else "one-shot"
        storage = "durable" if job.durable else "session"
        lines.append(
            f"{job.id}: {job.cron} -> {job.prompt[:60]} "
            f"[{frequency}, {storage}]"
        )
    return "\n".join(lines)


def run_cancel_cron(job_id: str) -> str:
    return cancel_job(job_id)


TOOLS.extend([
    {"name": "schedule_cron",
     "description": "Schedule a prompt with a 5-field cron expression.",
     "input_schema": {"type": "object",
                      "properties": {
                          "cron": {"type": "string"},
                          "prompt": {"type": "string"},
                          "recurring": {"type": "boolean"},
                          "durable": {"type": "boolean"}},
                      "required": ["cron", "prompt"]}},
    {"name": "list_crons", "description": "List scheduled cron jobs.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "cancel_cron", "description": "Cancel a cron job by ID.",
     "input_schema": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
])

TOOL_HANDLERS.update({
    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
})

# 执行工具
def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked is not None:
        return str(blocked)

    handler = TOOL_HANDLERS.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as error:
        output = f"Error: {error}"
    trigger_hooks("PostToolUse", block, output)
    return str(output)


# -- Scheduler and agent loop --

# Event 自己不碰线程，是线程的代码主动去查它。

# 退出时我们再 set() 成 True，线程下次醒来就自己结束。因为查的时候如果是true就是要停下了, 线程终止

RUNTIME_STOP = threading.Event() # 内部只有两个: 一个布尔标志 + 一份"正在等旗子的线程名单"
runtime_threads: list[threading.Thread] = []
runtime_started = False
runtime_lock = threading.Lock()
agent_lock = threading.Lock()
session_history: list = []

# 一秒检查一次有没有到时间
def cron_scheduler_loop(stop_event: threading.Event = RUNTIME_STOP):
    # stop_event初始是false, 线程创建之后循环一直看这个参数, 如果false就是一直执行(无人叫停), true就是执行完之后终止线程
    # # 注意：若被叫停时它正好在循环体里，会把【当前这一轮】干完，回到 wait 才发现
    while not stop_event.wait(1.0): 
        poll_due_jobs(datetime.now()) # 查现在时间, 如果到点了,写入磁盘, 将任务放进cron_queue


def agent_loop(messages: list, context: dict | None = None):
    fired = consume_cron_queue() # 拿的是可以直接执行(消费)的对象(一开始可能为空)
    scheduled_start = len(messages) #计划开始的下标, 回滚用 

    for job in fired: # 将任务提示词插入给模型, 模型会决定要不要调用工具, 调用哪个
        messages.append({"role": "user", "content": f"[Scheduled] {job.prompt}"})
        print(f"  [cron] delivered {job.id}: {job.prompt[:60]}")

    waiting_for_ack = list(fired) #防御性写法
    while True:
        try:
            response = client.messages.create(
                model=MODEL,
                system=SYSTEM,
                messages=messages,
                tools=TOOLS,
                max_tokens=8000,
            )
        except Exception as error: # 如果错误
            if waiting_for_ack:# 只有"有任务在投递、且还没确认"时才回滚
                del messages[scheduled_start:] # 撤掉刚注入的 [Scheduled] 消息（内存历史）
                restore_cron_jobs(waiting_for_ack)
            print(f"  [error] {type(error).__name__}: {error}")
            return context # 把上下文包打包回去

        messages.append({"role": "assistant", "content": response.content})
        if waiting_for_ack: # 如果有可以任务
            try:
                acknowledge_cron_jobs(waiting_for_ack) # 确认-> 状态转移 -> 落盘结果
            except Exception as error:
                print(f"  [cron] acknowledgement failed: {error}")
            waiting_for_ack = []

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return context

        results = []
        for block in tool_calls:
            output = execute_tool(block)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})


def print_latest_assistant_text(messages: list):
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            print(content)
        else:
            for block in content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    print(block.get("text", ""))
        return

# 锁中运行agent, 表示被占用
def run_agent_turn_locked(user_query: str | None = None):
    if user_query is not None:
        trigger_hooks("UserPromptSubmit", user_query)
        session_history.append({"role": "user", "content": user_query})
    agent_loop(session_history)
    print_latest_assistant_text(session_history)
    print()


def queue_processor_loop(stop_event: threading.Event = RUNTIME_STOP):
    while not stop_event.wait(0.2): # 每0.2秒检查队列中是否有东西
        if not has_cron_queue() or not agent_lock.acquire(blocking=False): # 有到期的定时任务任务, agent空闲
            continue
        try:
            if has_cron_queue():
                run_agent_turn_locked() # 运行有锁agentloop
        finally:
            agent_lock.release()

# 启动常驻的两条后台线程, 确保只启动一次
def start_runtime_threads(): 
    global runtime_started # 可以改全局变量 runtime_started
    with runtime_lock:
        if runtime_started: # 如果已经是True了就返回
            return
        load_durable_jobs() # 加载磁盘中的任务队列
        RUNTIME_STOP.clear() # 全局清理
        runtime_threads.extend([
            threading.Thread(
                target=cron_scheduler_loop, 
                name="cron-scheduler",
                daemon=True,
            ),
            threading.Thread(
                target=queue_processor_loop,
                name="cron-queue-processor",
                daemon=True,
            ),
        ])
        for thread in runtime_threads:
            thread.start() # 真正让线程开始跑
        runtime_started = True # 修改全局变量 runtime_started


def stop_runtime_threads():
    global runtime_started
    with runtime_lock:
        if not runtime_started:
            return
        RUNTIME_STOP.set()
        for thread in runtime_threads:
            thread.join(timeout=1)
        runtime_threads.clear()
        runtime_started = False


if __name__ == "__main__":
    print("s12: Cron Scheduler - run prompts on a local schedule")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    start_runtime_threads() # 两条线程一开始就起好了
    try:
        while True:
            try:
                # \001/\002 tell Readline the ANSI escapes have zero display width.
                query = input("\001\033[36m\002s12 >> \001\033[0m\002")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() in ("q", "exit", ""):
                break
            with agent_lock:
                run_agent_turn_locked(query) #
    finally:
        stop_runtime_threads()
