"""
AgentManager 技能测试

测试覆盖：
1. AgentProfile 创建 - dataclass 正确
2. Registry 注册/注销 - register/unregister/list
3. Registry 持久化 - save_to_file / load_from_file
4. PromptLoader 加载 - 正常加载、缓存、mtime 失效
5. PromptLoader 符号链接 - 解析真实路径
6. AgentSwitcher 切换 - switch_to / get_active
7. 模板变量解析 - {{name}} → 实际值
8. ManagerSkill 接口 - query / execute 标准接口
9. 热重载 - 文件修改后 prompt 重新加载
"""

import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_search.manager_skill import (
    AgentProfile,
    AgentRegistry,
    PromptLoader,
    AgentSwitcher,
    ManagerSkill,
    AGENT_MAP_FILE,
)


class TestAgentProfile(unittest.TestCase):
    """测试 AgentProfile dataclass"""

    def test_profile_creation(self):
        """AgentProfile 创建"""
        profile = AgentProfile(
            alias="test_agent",
            name="Test Agent",
            role="assistant",
            description="A test agent",
            prompt_path="prompts/test.txt",
            model_preference="gpt-4",
            temperature=0.8,
            max_tokens=8192
        )
        
        self.assertEqual(profile.alias, "test_agent")
        self.assertEqual(profile.name, "Test Agent")
        self.assertEqual(profile.role, "assistant")
        self.assertEqual(profile.description, "A test agent")
        self.assertEqual(profile.prompt_path, "prompts/test.txt")
        self.assertEqual(profile.model_preference, "gpt-4")
        self.assertEqual(profile.temperature, 0.8)
        self.assertEqual(profile.max_tokens, 8192)

    def test_profile_defaults(self):
        """AgentProfile 默认值"""
        profile = AgentProfile(
            alias="minimal",
            name="Minimal",
            role="assistant",
            description="",
            prompt_path="test.txt"
        )
        
        self.assertIsNone(profile.model_preference)
        self.assertEqual(profile.temperature, 0.7)
        self.assertEqual(profile.max_tokens, 4096)

    def test_to_dict_from_dict(self):
        """AgentProfile 序列化/反序列化"""
        original = AgentProfile(
            alias="serialize_test",
            name="Serialize Test",
            role="critic",
            description="Testing serialization",
            prompt_path="test.md",
            temperature=1.0
        )
        
        data = original.to_dict()
        restored = AgentProfile.from_dict(data)
        
        self.assertEqual(original.alias, restored.alias)
        self.assertEqual(original.name, restored.name)
        self.assertEqual(original.role, restored.role)
        self.assertEqual(original.temperature, restored.temperature)


class TestAgentRegistry(unittest.TestCase):
    """测试 AgentRegistry 注册表"""

    def setUp(self):
        """创建临时目录和注册表"""
        self.temp_dir = tempfile.mkdtemp()
        self.map_file = Path(self.temp_dir) / "agent_map.json"
        self.registry = AgentRegistry(map_file=self.map_file)

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register(self):
        """Registry 注册"""
        profile = AgentProfile(
            alias="assistant",
            name="Assistant",
            role="assistant",
            description="Default assistant",
            prompt_path="assistant.txt"
        )
        
        result = self.registry.register("assistant", profile)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["alias"], "assistant")
        self.assertEqual(self.registry.get("assistant"), profile)

    def test_unregister(self):
        """Registry 注销"""
        profile = AgentProfile(
            alias="temp",
            name="Temp",
            role="assistant",
            description="",
            prompt_path="temp.txt"
        )
        self.registry.register("temp", profile)
        
        result = self.registry.unregister("temp")
        
        self.assertTrue(result["success"])
        self.assertIsNone(self.registry.get("temp"))

    def test_unregister_nonexistent(self):
        """注销不存在的 Agent"""
        result = self.registry.unregister("nonexistent")
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_list_all(self):
        """列出所有 Agent"""
        agents = [
            AgentProfile("a1", "Agent 1", "assistant", "", "a1.txt"),
            AgentProfile("a2", "Agent 2", "critic", "", "a2.txt"),
            AgentProfile("a3", "Agent 3", "coordinator", "", "a3.txt"),
        ]
        for a in agents:
            self.registry.register(a.alias, a)
        
        result = self.registry.list_all()
        
        self.assertEqual(len(result), 3)
        aliases = {r["alias"] for r in result}
        self.assertEqual(aliases, {"a1", "a2", "a3"})


class TestRegistryPersistence(unittest.TestCase):
    """测试 Registry 持久化"""

    def setUp(self):
        """创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.map_file = Path(self.temp_dir) / "agent_map.json"

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load(self):
        """保存和加载"""
        registry = AgentRegistry(map_file=self.map_file)
        
        # 注册几个 agent
        profile1 = AgentProfile("agent1", "Agent 1", "assistant", "First", "a1.txt")
        profile2 = AgentProfile("agent2", "Agent 2", "critic", "Second", "a2.txt")
        registry.register("agent1", profile1)
        registry.register("agent2", profile2)
        
        # 保存
        save_result = registry.save_to_file()
        self.assertTrue(save_result["success"])
        self.assertTrue(self.map_file.exists())
        
        # 新建 registry 并加载
        registry2 = AgentRegistry(map_file=self.map_file)
        load_result = registry2.load_from_file()
        self.assertTrue(load_result["success"])
        self.assertEqual(load_result["loaded"], 2)
        
        # 验证
        agents = registry2.list_all()
        self.assertEqual(len(agents), 2)

    def test_load_nonexistent_file(self):
        """加载不存在的文件"""
        registry = AgentRegistry(map_file=Path(self.temp_dir) / "nonexistent.json")
        result = registry.load_from_file()
        self.assertTrue(result["success"])
        self.assertEqual(result["loaded"], 0)


class TestPromptLoader(unittest.TestCase):
    """测试 PromptLoader"""

    def setUp(self):
        """创建临时目录和 prompt 文件"""
        self.temp_dir = tempfile.mkdtemp()
        self.agent_dir = Path(self.temp_dir) / "agents"
        self.agent_dir.mkdir()
        self.loader = PromptLoader(self.agent_dir)

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_prompt(self):
        """加载 prompt 文件"""
        prompt_file = self.agent_dir / "hello.txt"
        prompt_file.write_text("Hello, {{name}}!")
        
        profile = AgentProfile(
            alias="greeter",
            name="Greeter",
            role="assistant",
            description="",
            prompt_path="hello.txt"
        )
        
        content = self.loader.load("greeter", profile)
        self.assertEqual(content, "Hello, {{name}}!")

    def test_cache(self):
        """缓存机制"""
        prompt_file = self.agent_dir / "cached.txt"
        prompt_file.write_text("Cached content")
        
        profile = AgentProfile(
            alias="cached",
            name="Cached",
            role="assistant",
            description="",
            prompt_path="cached.txt"
        )
        
        # 第一次加载
        content1 = self.loader.load("cached", profile)
        self.assertEqual(content1, "Cached content")
        
        # 修改文件
        time.sleep(0.1)  # 确保 mtime 不同
        prompt_file.write_text("Updated content")
        
        # 应该重新加载（因为 mtime 变了）
        content2 = self.loader.load("cached", profile)
        self.assertEqual(content2, "Updated content")

    def test_invalidate_cache(self):
        """手动使缓存失效"""
        prompt_file = self.agent_dir / "invalidate.txt"
        prompt_file.write_text("Original")
        
        profile = AgentProfile(
            alias="inv",
            name="Inv",
            role="assistant",
            description="",
            prompt_path="invalidate.txt"
        )
        
        self.loader.load("inv", profile)
        self.loader.invalidate("inv")
        
        # 再次加载应该读取文件
        time.sleep(0.1)
        prompt_file.write_text("New content")
        content = self.loader.load("inv", profile)
        self.assertEqual(content, "New content")

    def test_resolve_path_absolute(self):
        """解析绝对路径"""
        prompt_file = self.agent_dir / "absolute.txt"
        prompt_file.write_text("Absolute path")
        
        resolved = self.loader.resolve_path(str(prompt_file))
        self.assertEqual(resolved, prompt_file.resolve())

    def test_resolve_path_relative(self):
        """解析相对路径"""
        prompt_file = self.agent_dir / "relative.txt"
        prompt_file.write_text("Relative path")
        
        resolved = self.loader.resolve_path("relative.txt")
        self.assertEqual(resolved, prompt_file.resolve())


class TestSymlinkResolution(unittest.TestCase):
    """测试符号链接解析"""

    def setUp(self):
        """创建临时目录和符号链接"""
        self.temp_dir = tempfile.mkdtemp()
        self.agent_dir = Path(self.temp_dir) / "agents"
        self.agent_dir.mkdir()
        
        # 创建真实文件
        self.real_file = self.agent_dir / "real_prompt.txt"
        self.real_file.write_text("Real prompt content")
        
        # 创建符号链接
        self.link_file = self.agent_dir / "link_prompt.txt"
        if os.name != 'nt':  # Windows 不完全支持符号链接
            os.symlink(self.real_file, self.link_file)
        else:
            shutil.copy(self.real_file, self.link_file)
        
        self.loader = PromptLoader(self.agent_dir)

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_symlink(self):
        """解析符号链接"""
        resolved = self.loader.resolve_path("link_prompt.txt")
        # 解析后应该指向真实文件
        self.assertEqual(resolved.resolve(), self.real_file.resolve())


class TestAgentSwitcher(unittest.TestCase):
    """测试 AgentSwitcher"""

    def setUp(self):
        """创建临时目录和组件"""
        self.temp_dir = tempfile.mkdtemp()
        self.agent_dir = Path(self.temp_dir) / "agents"
        self.agent_dir.mkdir()
        
        # 创建 prompt 文件
        (self.agent_dir / "assistant.txt").write_text("You are {{name}}, an AI assistant.")
        (self.agent_dir / "critic.txt").write_text("You are a critic. Analyze: {{context.topic}}")
        
        # 创建注册表和加载器
        self.registry = AgentRegistry(map_file=Path(self.temp_dir) / "map.json")
        self.registry.register("assistant", AgentProfile(
            "assistant", "Assistant", "assistant", "", "assistant.txt"
        ))
        self.registry.register("critic", AgentProfile(
            "critic", "Critic", "critic", "", "critic.txt"
        ))
        
        self.loader = PromptLoader(self.agent_dir)
        self.switcher = AgentSwitcher(self.registry, self.loader)

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_switch_to(self):
        """切换 Agent"""
        result = self.switcher.switch_to("assistant")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["alias"], "assistant")
        self.assertEqual(result["role"], "assistant")
        self.assertIn("You are", result["prompt"])

    def test_switch_to_nonexistent(self):
        """切换到不存在的 Agent"""
        result = self.switcher.switch_to("nonexistent")
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_get_active(self):
        """获取当前活跃 Agent"""
        self.switcher.switch_to("assistant")
        
        active = self.switcher.get_active()
        self.assertTrue(active["active"])
        self.assertEqual(active["alias"], "assistant")

    def test_get_active_none(self):
        """没有活跃 Agent"""
        active = self.switcher.get_active()
        self.assertFalse(active["active"])

    def test_resolve_prompt_variables(self):
        """模板变量解析"""
        resolved = self.switcher.resolve_prompt("assistant", name="Alice")
        self.assertEqual(resolved, "You are Alice, an AI assistant.")

    def test_resolve_prompt_with_context(self):
        """context.key 语法"""
        resolved = self.switcher.resolve_prompt(
            "critic",
            context={"topic": "this text"}
        )
        self.assertEqual(resolved, "You are a critic. Analyze: this text")


class TestManagerSkillInterface(unittest.TestCase):
    """测试 ManagerSkill 标准接口"""

    def setUp(self):
        """创建临时目录和技能"""
        self.temp_dir = tempfile.mkdtemp()
        self.agent_dir = Path(self.temp_dir) / "agents"
        self.agent_dir.mkdir()
        
        self.skill = ManagerSkill(
            agent_dir=self.agent_dir,
            map_file=Path(self.temp_dir) / "map.json"
        )

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_query_register(self):
        """query: manager.register"""
        result = self.skill.query("manager.register", {
            "alias": "new_agent",
            "name": "New Agent",
            "role": "assistant",
            "description": "A new agent",
            "prompt_path": "new.txt"
        })
        
        self.assertTrue(result["success"])

    def test_query_switch(self):
        """query: manager.switch"""
        # 先注册
        (self.agent_dir / "test.txt").write_text("Test prompt")
        self.skill.execute("register", {"alias": "test", "prompt_path": "test.txt"})
        
        # 切换
        result = self.skill.query("manager.switch", {"alias": "test"})
        self.assertTrue(result["success"])

    def test_query_list(self):
        """query: manager.list"""
        result = self.skill.query("manager.list", {})
        self.assertTrue(result["success"])

    def test_query_unknown_capability(self):
        """query: 未知能力"""
        result = self.skill.query("manager.unknown", {})
        self.assertFalse(result["success"])
        self.assertIn("CAPABILITY_NOT_FOUND", result["error"]["code"])

    def test_execute_register(self):
        """execute: register"""
        (self.agent_dir / "exec_test.txt").write_text("Execute test")
        result = self.skill.execute("register", {
            "alias": "exec_agent",
            "prompt_path": "exec_test.txt"
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["alias"], "exec_agent")

    def test_execute_switch(self):
        """execute: switch"""
        (self.agent_dir / "switch.txt").write_text("Switch test")
        self.skill.execute("register", {"alias": "sw", "prompt_path": "switch.txt"})
        
        result = self.skill.execute("switch", {"alias": "sw"})
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["alias"], "sw")

    def test_execute_list(self):
        """execute: list"""
        result = self.skill.execute("list", {})
        self.assertTrue(result["success"])

    def test_execute_save(self):
        """execute: save"""
        result = self.skill.execute("save", {})
        self.assertTrue(result["success"])


class TestHotReload(unittest.TestCase):
    """测试热重载"""

    def setUp(self):
        """创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.agent_dir = Path(self.temp_dir) / "agents"
        self.agent_dir.mkdir()
        
        # 创建初始 prompt 文件
        self.prompt_file = self.agent_dir / "hot.txt"
        self.prompt_file.write_text("Initial content")
        
        self.skill = ManagerSkill(
            agent_dir=self.agent_dir,
            map_file=Path(self.temp_dir) / "map.json"
        )
        
        # 注册 agent
        self.skill.execute("register", {
            "alias": "hot",
            "name": "Hot Reload Agent",
            "prompt_path": "hot.txt"
        })

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reload_after_file_change(self):
        """文件修改后重新加载"""
        # 初始加载
        result1 = self.skill.execute("load_prompt", {"alias": "hot"})
        self.assertEqual(result1["data"]["prompt"], "Initial content")
        
        # 修改文件
        time.sleep(0.1)  # 确保 mtime 不同
        self.prompt_file.write_text("Updated content")
        
        # 触发热重载（invalidate）
        self.skill.notify("file_changed", {"alias": "hot"})
        
        # 重新加载
        result2 = self.skill.execute("load_prompt", {"alias": "hot"})
        self.assertEqual(result2["data"]["prompt"], "Updated content")

    def test_cache_invalidation(self):
        """缓存失效"""
        # 加载
        self.skill.execute("load_prompt", {"alias": "hot"})
        
        # 手动失效
        self.skill.loader.invalidate("hot")
        
        # 验证缓存已清
        self.assertNotIn("hot", self.skill.loader._cache)


class TestTemplateVariableParsing(unittest.TestCase):
    """测试模板变量解析"""

    def setUp(self):
        """创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.agent_dir = Path(self.temp_dir) / "agents"
        self.agent_dir.mkdir()
        
        self.skill = ManagerSkill(
            agent_dir=self.agent_dir,
            map_file=Path(self.temp_dir) / "map.json"
        )

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simple_variable(self):
        """简单变量替换"""
        (self.agent_dir / "simple.txt").write_text("Hello {{name}}!")
        self.skill.execute("register", {"alias": "simple", "prompt_path": "simple.txt"})
        
        result = self.skill.execute("resolve_prompt", {
            "alias": "simple",
            "name": "World"
        })
        
        self.assertEqual(result["data"]["resolved"], "Hello World!")

    def test_multiple_variables(self):
        """多个变量"""
        (self.agent_dir / "multi.txt").write_text("{{greeting}}, {{name}}!")
        self.skill.execute("register", {"alias": "multi", "prompt_path": "multi.txt"})
        
        result = self.skill.execute("resolve_prompt", {
            "alias": "multi",
            "greeting": "Hi",
            "name": "Alice"
        })
        
        self.assertEqual(result["data"]["resolved"], "Hi, Alice!")

    def test_context_key_syntax(self):
        """context.key 语法"""
        (self.agent_dir / "ctx.txt").write_text("Topic: {{context.topic}}")
        self.skill.execute("register", {"alias": "ctx", "prompt_path": "ctx.txt"})
        
        result = self.skill.execute("resolve_prompt", {
            "alias": "ctx",
            "context": {"topic": "AI Safety"}
        })
        
        self.assertEqual(result["data"]["resolved"], "Topic: AI Safety")


if __name__ == "__main__":
    unittest.main()
