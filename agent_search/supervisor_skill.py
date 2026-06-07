"""
supervisor_skill.py - AgentSupervisor 技能

监督、协调、监控子 Agent/子任务的执行

职责：
1. 任务协调（Orchestration）：管理复杂任务的工作流，支持子任务分解、依赖图、并行执行
2. 状态监控（Monitoring）：跟踪所有 active 任务/子 Agent 的状态，定期心跳检查
3. 异常恢复（Recovery）：任务失败时自动重试、fallback、死锁检测
4. 资源管理（Resource）：限制并发任务数、内存限制、超时管理
5. 进度报告（Reporting）：向用户报告任务进度百分比
"""

import asyncio
import time
import uuid
import logging
from typing import Any, Optional, Callable, Awaitable
from .skill_base import SkillBase
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class SupervisorConfig:
    """Supervisor 配置"""
    max_concurrent_tasks: int = 5      # 最大并发任务数
    task_timeout: int = 300            # 任务超时（秒）
    max_retries: int = 3              # 最大重试次数
    heartbeat_interval: int = 30       # 心跳间隔（秒）
    deadlock_threshold: int = 180      # 死锁阈值（秒）
    enable_monitoring: bool = True     # 启用状态监控


@dataclass
class Task:
    """任务单元"""
    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    retries: int = 0
    error: str | None = None
    result: Any | None = None
    dependencies: list[str] = field(default_factory=list)
    task_fn: Callable[[], Awaitable[Any]] | None = None  # 异步任务函数
    last_heartbeat: float = field(default_factory=time.time)
    fallback_fn: Callable[[], Any] | None = None  # 降级函数


@dataclass
class Workflow:
    """工作流定义"""
    workflow_id: str
    name: str
    steps: list[dict]  # [{"id": "...", "name": "...", "task_fn": ..., "deps": [...]}]
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING


class SupervisorSkill(SkillBase):
    """
    AgentSupervisor 技能
    
    监督、协调、监控子 Agent/子任务的执行
    """

    def __init__(self, config: SupervisorConfig | None = None):
        """初始化 Supervisor"""
        # 先解析 config，确保始终传递有效值给父类
        resolved_config = config or SupervisorConfig()
        
        # 初始化自己的属性
        self.config = resolved_config
        self._tasks: dict[str, Task] = {}
        self._workflows: dict[str, Workflow] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None
        self._shutdown = False
        
        # 事件回调
        self._event_handlers: dict[str, list[Callable]] = defaultdict(list)
        
        # 技能注册表（用于调用其他技能）
        self._skill_registry: dict[str, Any] = {}

        # 调用父类初始化（传递已解析的 config，避免被 None 覆盖）
        super().__init__(resolved_config)

    # ==================== 标准 Skill 接口 ====================
    def query(self, capability: str, context: dict | None = None) -> dict:
        """
        查询技能能力
        
        Args:
            capability: 能力名称
            context: 上下文信息
        
        Returns:
            能力描述字典
        """
        capabilities = {
            "orchestrate": {
                "description": "创建和管理复杂任务工作流",
                "params": ["name", "steps"]
            },
            "monitor": {
                "description": "监控任务执行状态",
                "params": []
            },
            "progress": {
                "description": "获取任务进度百分比",
                "params": []
            },
            "retry": {
                "description": "重试失败任务",
                "params": ["task_id"]
            },
            "cancel": {
                "description": "取消任务",
                "params": ["task_id"]
            },
            "status": {
                "description": "获取特定任务状态",
                "params": ["task_id"]
            },
        }
        
        if capability == "list":
            return {"capabilities": list(capabilities.keys())}
        
        return capabilities.get(capability, {"error": f"未知能力: {capability}"})

    def execute(self, action: str, params: dict | None = None) -> dict:
        """
        执行动作
        
        Args:
            action: 动作名称
            params: 参数
        
        Returns:
            执行结果
        """
        params = params or {}
        
        if action == "orchestrate":
            # 创建工作流
            workflow_id = self.create_workflow(
                name=params.get("name", "unnamed"),
                steps=params.get("steps", [])
            )
            return {"workflow_id": workflow_id, "status": "started"}
        
        elif action == "progress":
            # 获取进度
            return self.get_progress()
        
        elif action == "status":
            # 获取任务状态
            task_id = params.get("task_id")
            return self.get_task_status(task_id)
        
        elif action == "list_tasks":
            # 列出任务
            return {"tasks": self.list_tasks(params.get("status_filter"))}
        
        elif action == "cancel":
            # 取消任务
            task_id = params.get("task_id")
            return self.cancel_task(task_id)
        
        elif action == "retry":
            # 重试任务
            task_id = params.get("task_id")
            return self.retry_task(task_id)
        
        elif action == "start_monitoring":
            # 启动监控循环
            asyncio.create_task(self.monitor_loop())
            return {"status": "monitoring_started"}
        
        elif action == "stop_monitoring":
            # 停止监控循环
            self._shutdown = True
            if self._monitor_task:
                self._monitor_task.cancel()
            return {"status": "monitoring_stopped"}
        
        else:
            return {"error": f"未知动作: {action}"}

    def notify(self, event: str, data: dict | None = None):
        """
        通知事件
        
        Args:
            event: 事件名称
            data: 事件数据
        """
        data = data or {}
        logger.info(f"收到事件: {event}, 数据: {data}")
        
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(data))
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"事件处理器错误: {e}")

    # V 21:52 SkillBase delegation (V 反思 SOP 第 10 件加强版: util 化)
    def _handle_query(self, capability: str, context: dict) -> dict:
        return self.query(capability, context)

    def _handle_execute(self, action: str, params: dict) -> dict:
        return self.execute(action, params)

    def register_skill(self, name: str, skill: Any):
        """注册其他技能以便调用"""
        self._skill_registry[name] = skill

    def on(self, event: str, handler: Callable):
        """注册事件处理器"""
        self._event_handlers[event].append(handler)

    # ==================== 核心功能 ====================

    def create_task(
        self,
        name: str,
        task_fn: Callable[[], Awaitable[Any]],
        dependencies: list[str] = None,
        fallback_fn: Callable[[], Any] | None = None
    ) -> str:
        """
        创建任务
        
        Args:
            name: 任务名称
            task_fn: 异步任务函数
            dependencies: 依赖的任务ID列表
            fallback_fn: 降级函数
        
        Returns:
            task_id
        """
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            task_id=task_id,
            name=name,
            task_fn=task_fn,
            dependencies=dependencies or [],
            fallback_fn=fallback_fn
        )
        self._tasks[task_id] = task
        logger.info(f"创建任务: {task_id} - {name}")
        return task_id

    def get_task_status(self, task_id: str | None) -> dict:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
        
        Returns:
            状态字典
        """
        if not task_id:
            return {"error": "task_id 不能为空"}
        task = self._tasks.get(task_id)
        if not task:
            return {"error": f"任务不存在: {task_id}"}
        
        elapsed = 0
        if task.started_at:
            elapsed = time.time() - task.started_at
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "retries": task.retries,
            "error": task.error,
            "result": task.result,
            "elapsed": elapsed,
            "dependencies": task.dependencies,
        }

    def list_tasks(self, status_filter: TaskStatus | None = None) -> list[dict]:
        """
        列出所有任务
        
        Args:
            status_filter: 状态过滤器
        
        Returns:
            任务列表
        """
        tasks = self._tasks.values()
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        
        return [self.get_task_status(t.task_id) for t in tasks]

    def cancel_task(self, task_id: str) -> dict:
        """
        取消任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            结果字典
        """
        task = self._tasks.get(task_id)
        if not task:
            return {"error": f"任务不存在: {task_id}"}
        
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            return {"error": f"任务已结束，无法取消: {task.status.value}"}
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        logger.info(f"任务已取消: {task_id}")
        
        return {"task_id": task_id, "status": "cancelled"}

    def retry_task(self, task_id: str) -> dict:
        """
        重试任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            结果字典
        """
        task = self._tasks.get(task_id)
        if not task:
            return {"error": f"任务不存在: {task_id}"}
        
        if task.status != TaskStatus.FAILED:
            return {"error": f"只有失败任务可以重试，当前状态: {task.status.value}"}
        
        task.status = TaskStatus.PENDING
        task.retries = 0
        task.error = None
        logger.info(f"任务重置为待执行: {task_id}")
        
        return {"task_id": task_id, "status": "pending"}

    def get_progress(self) -> dict:
        """
        获取整体进度
        
        Returns:
            进度字典
        """
        tasks = list(self._tasks.values())
        total = len(tasks)
        
        if total == 0:
            return {
                "total": 0,
                "completed": 0,
                "running": 0,
                "pending": 0,
                "failed": 0,
                "percent": 0.0,
                "active_tasks": [],
                "failed_tasks": [],
            }
        
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        running = sum(1 for t in tasks if t.status == TaskStatus.RUNNING)
        pending = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        retrying = sum(1 for t in tasks if t.status == TaskStatus.RETRYING)
        
        active_tasks = []
        for t in tasks:
            if t.status == TaskStatus.RUNNING:
                elapsed = time.time() - t.started_at if t.started_at else 0
                active_tasks.append({
                    "task_id": t.task_id,
                    "name": t.name,
                    "status": t.status.value,
                    "elapsed": elapsed,
                })
        
        failed_tasks = []
        for t in tasks:
            if t.status == TaskStatus.FAILED:
                failed_tasks.append({
                    "task_id": t.task_id,
                    "name": t.name,
                    "error": t.error,
                })
        
        # 计算百分比（基于已完成和进行中的任务）
        in_progress = completed + running
        percent = (in_progress / total) * 100 if total > 0 else 0.0
        
        return {
            "total": total,
            "completed": completed,
            "running": running,
            "pending": pending,
            "failed": failed,
            "retrying": retrying,
            "percent": round(percent, 1),
            "active_tasks": active_tasks,
            "failed_tasks": failed_tasks,
        }

    # ==================== 工作流管理 ====================

    def create_workflow(self, name: str, steps: list[dict]) -> str:
        """
        创建工作流（多个有依赖关系的任务）
        
        Args:
            name: 工作流名称
            steps: 步骤列表
                [
                    {"id": "step1", "name": "搜索", "task_fn": lambda: search(...), "deps": []},
                    {"id": "step2", "name": "分析", "task_fn": lambda: analyze(...), "deps": ["step1"]},
                    {"id": "step3", "name": "报告", "task_fn": lambda: report(...), "deps": ["step2"]},
                ]
        
        Returns:
            workflow_id
        """
        workflow_id = str(uuid.uuid4())[:8]
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            steps=steps
        )
        self._workflows[workflow_id] = workflow
        logger.info(f"创建工作流: {workflow_id} - {name}")
        
        # 启动工作流执行
        asyncio.create_task(self._execute_workflow(workflow))
        
        return workflow_id

    async def _execute_workflow(self, workflow: Workflow):
        """
        执行工作流
        
        Args:
            workflow: 工作流对象
        """
        logger.info(f"开始执行工作流: {workflow.workflow_id}")
        
        # 拓扑排序，确定执行顺序
        sorted_steps = self._topological_sort(workflow.steps)
        
        # 按顺序执行任务（尊重依赖关系，可并行的会并行执行）
        await self._execute_ordered_steps(sorted_steps, workflow.steps)
        
        logger.info(f"工作流执行完成: {workflow.workflow_id}")

    def _topological_sort(self, steps: list[dict]) -> list[list[str]]:
        """
        拓扑排序，返回可以并行执行的步骤分组
        
        Args:
            steps: 步骤列表
        
        Returns:
            分层列表 [["step1", "step2"], ["step3"], ["step4"]] 表示可并行执行的批次
        """
        step_ids = {s["id"] for s in steps}
        deps_map = {s["id"]: set(s.get("deps", [])) for s in steps}
        
        # 验证依赖
        for step_id, deps in deps_map.items():
            invalid = deps - step_ids
            if invalid:
                raise ValueError(f"步骤 {step_id} 依赖不存在的步骤: {invalid}")
        
        result = []
        remaining = set(step_ids)
        completed = set()
        
        while remaining:
            # 找出所有依赖都已完成的步骤
            ready = {
                sid for sid in remaining
                if all(dep in completed for dep in deps_map[sid])
            }
            
            if not ready:
                raise ValueError("工作流存在循环依赖")
            
            result.append(list(ready))
            completed.update(ready)
            remaining -= ready
        
        return result

    async def _execute_ordered_steps(self, sorted_steps: list[list[str]], all_steps: list[dict]):
        """
        按拓扑顺序执行步骤
        
        Args:
            sorted_steps: 分层排序后的步骤ID列表
            all_steps: 所有步骤信息
        """
        step_map = {s["id"]: s for s in all_steps}
        task_id_map = {}  # step_id -> task_id
        
        for batch in sorted_steps:
            # 并行执行同一批次的任务
            tasks = []
            for step_id in batch:
                step = step_map[step_id]
                task_fn = step.get("task_fn")
                
                # 获取依赖的 task_id
                deps = [task_id_map[d] for d in step.get("deps", []) if d in task_id_map]
                
                if task_fn:
                    task_id = self.create_task(
                        name=step.get("name", step_id),
                        task_fn=task_fn,
                        dependencies=deps
                    )
                else:
                    # 如果没有 task_fn，创建一个空任务
                    async def noop():
                        return {"step": step_id, "result": "completed"}
                    task_id = self.create_task(
                        name=step.get("name", step_id),
                        task_fn=noop,
                        dependencies=deps
                    )
                
                task_id_map[step_id] = task_id
                tasks.append(self._run_task(task_id))
            
            # 等待这一批所有任务完成
            await asyncio.gather(*tasks, return_exceptions=True)

    # ==================== 任务执行 ====================

    async def _run_task(self, task_id: str):
        """
        运行单个任务
        
        Args:
            task_id: 任务ID
        """
        task = self._tasks.get(task_id)
        if not task:
            return
        
        # 检查依赖是否都已完成
        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                task.status = TaskStatus.FAILED
                task.error = f"依赖任务 {dep_id} 未完成"
                return
        
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.last_heartbeat = time.time()
            logger.info(f"任务开始执行: {task_id} - {task.name}")
            
            try:
                # 执行带超时的任务
                result = await asyncio.wait_for(
                    task.task_fn(),
                    timeout=self.config.task_timeout
                )
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                logger.info(f"任务完成: {task_id} - {task.name}")
                
            except asyncio.TimeoutError:
                task.error = f"任务超时 ({self.config.task_timeout}秒)"
                await self._handle_task_failure(task)
                
            except Exception as e:
                task.error = str(e)
                await self._handle_task_failure(task)

    async def _handle_task_failure(self, task: Task):
        """
        处理任务失败
        
        Args:
            task: 失败的任务
        """
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        logger.warning(f"任务失败: {task.task_id} - {task.name}: {task.error}")
        
        # 尝试重试
        if self._should_retry(task):
            task.status = TaskStatus.RETRYING
            task.retries += 1
            logger.info(f"任务将在延迟后重试: {task.task_id} (第 {task.retries} 次)")
            
            await asyncio.sleep(2 ** task.retries)  # 指数退避
            asyncio.create_task(self._run_task(task.task_id))
        else:
            # 使用降级策略
            if task.fallback_fn:
                try:
                    task.result = task.fallback_fn()
                    task.status = TaskStatus.COMPLETED
                    logger.info(f"任务使用降级策略恢复: {task.task_id}")
                except Exception as e:
                    logger.error(f"降级策略失败: {task.task_id}: {e}")

    def _should_retry(self, task: Task) -> bool:
        """
        判断任务是否应该重试
        
        Args:
            task: 任务对象
        
        Returns:
            是否重试
        """
        if task.retries >= self.config.max_retries:
            return False
        
        # 不可重试的错误类型
        non_retryable = ["超时", "timeout", "cancelled", "取消"]
        if task.error:
            for pattern in non_retryable:
                if pattern.lower() in task.error.lower():
                    return False
        
        return True

    # ==================== 监控循环 ====================

    async def monitor_loop(self):
        """
        监控循环（后台运行）
        
        定期检查：
        1. 任务心跳
        2. 死锁检测
        3. 超时任务
        """
        self._monitoring = True
        logger.info("监控循环启动")
        
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if not self.config.enable_monitoring:
                    continue
                
                # 死锁检测
                deadlocked = self._detect_deadlock()
                for task_id in deadlocked:
                    task = self._tasks.get(task_id)
                    if task:
                        task.error = f"死锁检测：任务超过 {self.config.deadlock_threshold} 秒无响应"
                        await self._handle_task_failure(task)
                
                # 更新心跳
                for task in self._tasks.values():
                    if task.status == TaskStatus.RUNNING:
                        task.last_heartbeat = time.time()
                
                # 发送心跳事件
                progress = self.get_progress()
                self.notify("heartbeat", {"progress": progress})
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
        
        self._monitoring = False
        logger.info("监控循环停止")

    def _detect_deadlock(self) -> list[str]:
        """
        检测死锁任务
        
        Returns:
            死锁任务ID列表
        """
        deadlocked = []
        now = time.time()
        
        for task in self._tasks.values():
            if task.status == TaskStatus.RUNNING:
                elapsed = now - task.last_heartbeat
                if elapsed > self.config.deadlock_threshold:
                    deadlocked.append(task.task_id)
        
        return deadlocked

    # ==================== 上下文管理器 ====================

    @asynccontextmanager
    async def session(self):
        """
        创建 Supervisor 会话
        
        用法:
            async with supervisor.session():
                # 使用 supervisor
        """
        try:
            yield self
        finally:
            self._shutdown = True
            if self._monitor_task:
                self._monitor_task.cancel()


# ==================== 便捷函数 ====================

_default_supervisor: SupervisorSkill | None = None


def get_supervisor(config: SupervisorConfig | None = None) -> SupervisorSkill:
    """
    获取默认 Supervisor 实例
    
    Args:
        config: 配置
    
    Returns:
        SupervisorSkill 实例
    """
    global _default_supervisor
    if _default_supervisor is None:
        _default_supervisor = SupervisorSkill(config)
    return _default_supervisor
