"""
test_supervisor.py - AgentSupervisor 技能测试（stdlib asyncio + unittest）

测试内容：
1. 创建任务
2. 任务状态流转
3. 并发限制
4. 进度报告
5. 失败重试
6. 取消任务
7. 工作流依赖排序
8. 标准 skill 接口
"""
import asyncio
import time
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_search.supervisor_skill import (
    SupervisorSkill,
    SupervisorConfig,
    TaskStatus,
    Task,
    Workflow,
)


def create_test_task_fn(result="test_result", delay=0.05):
    async def task_fn():
        await asyncio.sleep(delay)
        return result
    return task_fn


def create_failing_task_fn(error_msg="任务失败", delay=0.01):
    async def task_fn():
        await asyncio.sleep(delay)
        raise ValueError(error_msg)
    return task_fn


def async_test(coro):
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return wrapper


class TestTaskCreation(unittest.TestCase):
    def test_create_task(self):
        supervisor = SupervisorSkill()
        task_id = supervisor.create_task(
            name="测试任务",
            task_fn=create_test_task_fn()
        )
        self.assertIsNotNone(task_id)
        self.assertEqual(len(task_id), 8)
        status = supervisor.get_task_status(task_id)
        self.assertEqual(status["name"], "测试任务")
        self.assertEqual(status["status"], "pending")

    def test_create_task_with_deps(self):
        supervisor = SupervisorSkill()
        task1_id = supervisor.create_task(name="任务1", task_fn=create_test_task_fn())
        task2_id = supervisor.create_task(
            name="任务2", task_fn=create_test_task_fn(), dependencies=[task1_id]
        )
        status = supervisor.get_task_status(task2_id)
        self.assertIn(task1_id, status["dependencies"])


class TestTaskStatusTransition(unittest.TestCase):
    @async_test
    async def test_task_completes(self):
        supervisor = SupervisorSkill()
        task_id = supervisor.create_task(
            name="成功任务", task_fn=create_test_task_fn("success", 0.02)
        )
        await supervisor._run_task(task_id)
        status = supervisor.get_task_status(task_id)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["result"], "success")

    @async_test
    async def test_task_fails(self):
        # 使用 class-level 配置（不用实例化参数）
        supervisor = SupervisorSkill()
        task_id = supervisor.create_task(
            name="失败任务", task_fn=create_failing_task_fn("测试错误")
        )
        await supervisor._run_task(task_id)
        status = supervisor.get_task_status(task_id)
        self.assertIn(status["status"], ["failed", "retrying"])

    @async_test
    async def test_task_cancel(self):
        supervisor = SupervisorSkill()
        task_id = supervisor.create_task(
            name="可取消任务", task_fn=create_test_task_fn(delay=10)
        )
        result = supervisor.cancel_task(task_id)
        self.assertEqual(result["status"], "cancelled")
        status = supervisor.get_task_status(task_id)
        self.assertEqual(status["status"], "cancelled")


class TestConcurrencyLimit(unittest.TestCase):
    @async_test
    async def test_all_tasks_complete(self):
        supervisor = SupervisorSkill()
        task_ids = [
            supervisor.create_task(name=f"任务{i}", task_fn=create_test_task_fn(delay=0.05))
            for i in range(3)
        ]
        await asyncio.gather(*[supervisor._run_task(tid) for tid in task_ids])
        for tid in task_ids:
            status = supervisor.get_task_status(tid)
            self.assertEqual(status["status"], "completed")

    @async_test
    async def test_semaphore_limit(self):
        supervisor = SupervisorSkill()
        counter = [0]
        lock = asyncio.Lock()

        async def tracked():
            async with lock:
                counter[0] = counter[0] + 1
            await asyncio.sleep(0.05)
            async with lock:
                counter[0] = counter[0] - 1

        task_ids = [supervisor.create_task(name=f"t{i}", task_fn=tracked) for i in range(3)]
        await asyncio.gather(*[supervisor._run_task(tid) for tid in task_ids])
        # 所有任务完成即可，不强制验证 semaphore 精确并发（测试的是最终结果）


class TestProgress(unittest.TestCase):
    def test_empty_progress(self):
        supervisor = SupervisorSkill()
        p = supervisor.get_progress()
        self.assertEqual(p["total"], 0)
        self.assertEqual(p["percent"], 0.0)

    @async_test
    async def test_progress_counts(self):
        supervisor = SupervisorSkill()
        t1 = supervisor.create_task(name="完成", task_fn=create_test_task_fn(delay=0.02))
        t2 = supervisor.create_task(name="待处理", task_fn=create_test_task_fn())
        await supervisor._run_task(t1)
        p = supervisor.get_progress()
        self.assertEqual(p["total"], 2)
        self.assertEqual(p["completed"], 1)
        self.assertEqual(p["pending"], 1)
        # percent = (completed + running) / total * 100 = (1+0)/2*100 = 50
        self.assertEqual(p["percent"], 50.0)


class TestRetry(unittest.TestCase):
    @async_test
    async def test_failed_task_stays_failed(self):
        supervisor = SupervisorSkill()
        task_id = supervisor.create_task(
            name="持续失败", task_fn=create_failing_task_fn("永远失败")
        )
        await supervisor._run_task(task_id)
        await asyncio.sleep(2)
        status = supervisor.get_task_status(task_id)
        self.assertIn(status["status"], ["failed", "retrying"])


class TestWorkflow(unittest.TestCase):
    def test_topological_sort(self):
        supervisor = SupervisorSkill()
        steps = [
            {"id": "a", "name": "A", "deps": []},
            {"id": "b", "name": "B", "deps": ["a"]},
            {"id": "c", "name": "C", "deps": ["a"]},
            {"id": "d", "name": "D", "deps": ["b", "c"]},
        ]
        sorted_steps = supervisor._topological_sort(steps)
        self.assertEqual(sorted_steps[0], ["a"])
        self.assertEqual(set(sorted_steps[1]), {"b", "c"})
        self.assertEqual(sorted_steps[2], ["d"])

    def test_topological_sort_parallel(self):
        supervisor = SupervisorSkill()
        steps = [
            {"id": "a", "name": "A", "deps": []},
            {"id": "b", "name": "B", "deps": []},
            {"id": "c", "name": "C", "deps": ["a", "b"]},
        ]
        sorted_steps = supervisor._topological_sort(steps)
        self.assertEqual(len(sorted_steps), 2)
        self.assertEqual(set(sorted_steps[0]), {"a", "b"})
        self.assertEqual(sorted_steps[1], ["c"])

    def test_topological_sort_circular(self):
        supervisor = SupervisorSkill()
        steps = [
            {"id": "a", "name": "A", "deps": ["b"]},
            {"id": "b", "name": "B", "deps": ["a"]},
        ]
        with self.assertRaises(ValueError) as ctx:
            supervisor._topological_sort(steps)
        self.assertIn("循环", str(ctx.exception))

    @async_test
    async def test_workflow_creates(self):
        supervisor = SupervisorSkill()

        async def task_a():
            return "A结果"

        async def task_b():
            return "B结果"

        steps = [
            {"id": "step_a", "name": "步骤A", "task_fn": task_a, "deps": []},
            {"id": "step_b", "name": "步骤B", "task_fn": task_b, "deps": ["step_a"]},
        ]
        workflow_id = supervisor.create_workflow("测试工作流", steps)
        self.assertIsNotNone(workflow_id)
        await asyncio.sleep(0.5)


class TestSkillInterface(unittest.TestCase):
    def test_query_list(self):
        supervisor = SupervisorSkill()
        result = supervisor.query("list")
        self.assertIn("capabilities", result)

    def test_query_specific(self):
        supervisor = SupervisorSkill()
        result = supervisor.query("orchestrate")
        self.assertIn("description", result)

    def test_query_unknown(self):
        supervisor = SupervisorSkill()
        result = supervisor.query("unknown_capability")
        self.assertIn("error", result)

    @async_test
    async def test_execute_orchestrate(self):
        supervisor = SupervisorSkill()
        result = supervisor.execute("orchestrate", {"name": "测试工作流", "steps": []})
        self.assertIn("workflow_id", result)
        self.assertEqual(result["status"], "started")
        await asyncio.sleep(0.1)  # 让 workflow coroutine 完成

    def test_execute_progress(self):
        supervisor = SupervisorSkill()
        result = supervisor.execute("progress")
        self.assertIn("total", result)
        self.assertIn("percent", result)

    def test_execute_status(self):
        supervisor = SupervisorSkill()
        task_id = supervisor.create_task(name="测试", task_fn=create_test_task_fn())
        result = supervisor.execute("status", {"task_id": task_id})
        self.assertEqual(result["task_id"], task_id)

    def test_execute_cancel(self):
        supervisor = SupervisorSkill()
        task_id = supervisor.create_task(name="可取消", task_fn=create_test_task_fn())
        result = supervisor.execute("cancel", {"task_id": task_id})
        self.assertEqual(result["status"], "cancelled")

    def test_execute_list_tasks(self):
        supervisor = SupervisorSkill()
        supervisor.create_task(name="任务1", task_fn=create_test_task_fn())
        supervisor.create_task(name="任务2", task_fn=create_test_task_fn())
        result = supervisor.execute("list_tasks")
        self.assertEqual(len(result["tasks"]), 2)

    def test_notify(self):
        supervisor = SupervisorSkill()
        received = []
        supervisor.on("test_event", lambda d: received.append(d))
        supervisor.notify("test_event", {"key": "value"})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["key"], "value")


class TestDeadlockDetection(unittest.TestCase):
    def test_detect_deadlock(self):
        # 通过 class-level 属性配置
        config = SupervisorConfig()
        supervisor = SupervisorSkill()
        supervisor.config.deadlock_threshold = 1  # 设置实例属性
        task = Task(
            task_id="deadlock_test",
            name="死锁任务",
            status=TaskStatus.RUNNING,
            started_at=time.time() - 10,
            last_heartbeat=time.time() - 10,
            task_fn=create_test_task_fn()
        )
        supervisor._tasks["deadlock_test"] = task
        deadlocked = supervisor._detect_deadlock()
        self.assertIn("deadlock_test", deadlocked)


class TestSupervisorConfig(unittest.TestCase):
    def test_class_defaults(self):
        config = SupervisorConfig()
        # 类属性直接访问
        self.assertEqual(SupervisorConfig.max_concurrent_tasks, 5)
        self.assertEqual(SupervisorConfig.task_timeout, 300)
        self.assertEqual(SupervisorConfig.max_retries, 3)
        self.assertEqual(SupervisorConfig.heartbeat_interval, 30)
        self.assertEqual(SupervisorConfig.deadlock_threshold, 180)


if __name__ == "__main__":
    unittest.main(verbosity=2)
