"""
AgentManager 技能 - 管理多个 Agent 的 prompt 模板

职责：
1. Agent 注册表（别名→配置的映射，持久化到 agent_map.json）
2. Prompt 加载器（支持热重载、符号链接、缓存）
3. Agent 切换器（角色切换、模板变量解析）
4. 与 AgentSymphony 生态集成

与 VCP agentManager.js 的差异：
- 平台：Node.js → Python
- 文件格式：纯 JSON → JSON + Python dataclass
- 目录扫描：chokidar → pathlib
- 符号链接：os.path.realpath
- 缓存：Map + 文件监听 → LRU 缓存 + mtime 检测
- 模板变量：{{variable}} + {{context.key}}
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Any, Optional
from .skill_base import SkillBase
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# 默认 agent_map.json 路径
AGENT_MAP_FILE = Path.home() / ".agent-search" / "agent_map.json"


# ==================== AgentProfile ====================

@dataclass
class AgentProfile:
    """单个 Agent 配置"""
    alias: str
    name: str
    role: str  # "assistant" / "critic" / "coordinator"
    description: str
    prompt_path: str  # 相对路径或绝对路径
    model_preference: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentProfile":
        return cls(**data)


# ==================== AgentRegistry ====================

class AgentRegistry(SkillBase):
    """Agent 注册表（内存缓存 + JSON 持久化）"""

    def __init__(self, map_file: Optional[Path] = None):
        self.map_file = map_file or AGENT_MAP_FILE
        self._agents: dict[str, AgentProfile] = {}
        # 确保目录存在
        self.map_file.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(map_file)

    # V 21:52 SkillBase delegation (V 反思 SOP 第 10 件加强版: util 化)
    # Fix S1 (Remaining-Components-Audit): _handle_query / _handle_execute 不再调
    # self.query / self.execute（那样会无限递归——SkillBase.query 调用 _handle_query），
    # 改为直接走原 capability_map / if-elif 逻辑。
    def _handle_query(self, capability: str, context: dict) -> dict:
        context = context or {}
        capability_map = {
            "manager.register": self._do_register,
            "manager.switch": self._do_switch,
            "manager.get_active": self._do_get_active,
            "manager.list": self._do_list,
            "manager.reload": self._do_reload,
            "manager.unregister": self._do_unregister,
            "manager.load_prompt": self._do_load_prompt,
        }
        if capability not in capability_map:
            return {"success": False, "error": f"未知 capability: {capability}"}
        return capability_map[capability](context)

    def _handle_execute(self, action: str, params: dict) -> dict:
        if action == "register":
            return self._do_register(params)
        if action == "switch":
            return self._do_switch(params)
        if action == "get_active":
            return self._do_get_active(params)
        if action == "list":
            return self._do_list(params)
        if action == "reload":
            return self._do_reload(params)
        if action == "unregister":
            return self._do_unregister(params)
        if action == "load_prompt":
            return self._do_load_prompt(params)
        return {"success": False, "error": f"未知 action: {action}"}
    def register(self, alias: str, profile: AgentProfile) -> dict:
        """注册 Agent，返回注册结果"""
        if not isinstance(profile, AgentProfile):
            return {"success": False, "error": "profile 必须是 AgentProfile 实例"}
        
        if profile.alias != alias:
            profile.alias = alias  # 确保 alias 一致
        
        self._agents[alias] = profile
        logger.info(f"注册 Agent: {alias} ({profile.role})")
        
        return {
            "success": True,
            "alias": alias,
            "profile": profile.to_dict()
        }

    def unregister(self, alias: str) -> dict:
        """注销 Agent"""
        if alias not in self._agents:
            return {"success": False, "error": f"Agent 不存在: {alias}"}
        
        profile = self._agents.pop(alias)
        logger.info(f"注销 Agent: {alias}")
        
        return {
            "success": True,
            "alias": alias,
            "removed_profile": profile.to_dict()
        }

    def get(self, alias: str) -> Optional[AgentProfile]:
        """获取 Agent 配置"""
        return self._agents.get(alias)

    def list_all(self) -> list[dict]:
        """列出所有 Agent"""
        return [
            {"alias": alias, **profile.to_dict()}
            for alias, profile in self._agents.items()
        ]

    def save_to_file(self) -> dict:
        """持久化到 agent_map.json"""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "agents": {
                    alias: profile.to_dict()
                    for alias, profile in self._agents.items()
                }
            }
            with open(self.map_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"保存 Agent 注册表到: {self.map_file}")
            return {"success": True, "file": str(self.map_file)}
        except Exception as e:
            logger.error(f"保存失败: {e}")
            return {"success": False, "error": str(e)}

    def load_from_file(self) -> dict:
        """从 agent_map.json 加载"""
        if not self.map_file.exists():
            logger.info(f"文件不存在，跳过加载: {self.map_file}")
            return {"success": True, "loaded": 0}

        try:
            with open(self.map_file, encoding="utf-8") as f:
                data = json.load(f)
            
            agents_data = data.get("agents", {})
            count = 0
            for alias, profile_data in agents_data.items():
                try:
                    profile = AgentProfile.from_dict(profile_data)
                    self._agents[alias] = profile
                    count += 1
                except Exception as e:
                    logger.warning(f"跳过无效 Agent {alias}: {e}")
            
            logger.info(f"加载了 {count} 个 Agent")
            return {"success": True, "loaded": count}
        except Exception as e:
            logger.error(f"加载失败: {e}")
            return {"success": False, "error": str(e)}


# ==================== PromptLoader ====================

class PromptLoader:
    """动态加载 Agent prompt 文件，支持热重载"""

    def __init__(self, agent_dir: Path):
        self.agent_dir = Path(agent_dir)
        self._cache: dict[str, str] = {}  # alias -> prompt content
        self._mtime: dict[str, float] = {}  # alias -> file mtime
        self._watcher: Optional[Any] = None  # 监听器

    def resolve_path(self, prompt_path: str) -> Path:
        """
        解析 prompt 路径，支持符号链接
        
        Args:
            prompt_path: 相对路径或绝对路径
        
        Returns:
            解析后的 Path 对象
        """
        path = Path(prompt_path)
        
        # 如果是绝对路径，直接返回（解析符号链接）
        if path.is_absolute():
            return path.resolve()
        
        # 相对路径：先在 agent_dir 中查找
        full_path = self.agent_dir / path
        if full_path.exists():
            return full_path.resolve()
        
        # 兜底：直接 resolve（相对路径相对于当前目录）
        return path.resolve()

    def load(self, alias: str, profile: AgentProfile) -> str:
        """
        加载 prompt，支持缓存和热重载
        
        Args:
            alias: Agent 别名
            profile: AgentProfile 配置
        
        Returns:
            prompt 文本内容
        
        Fix: TOCTOU 修复 - 先打开文件获取内容，再检查/更新缓存，
        避免 exists()+stat()+open() 三步之间的文件变化导致的问题。
        使用 try/except 捕获文件缺失错误作为缓存未命中的兜底。
        """
        try:
            prompt_path = self.resolve_path(profile.prompt_path)

            # 获取当前文件修改时间（用于更新缓存时间戳）
            current_mtime = prompt_path.stat().st_mtime

            # 检查缓存是否有效（文件未修改）
            if alias in self._cache:
                if alias in self._mtime and self._mtime[alias] >= current_mtime:
                    logger.debug(f"从缓存加载 prompt: {alias}")
                    return self._cache[alias]

            # 重新加载（先读内容，后更新缓存）
            # 修复 TOCTOU: 直接 open 读取，让 OS 处理文件缺失情况
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                logger.warning(f"Prompt 文件已被删除或移动: {prompt_path}")
                return f"[Prompt 文件不存在: {prompt_path}]"

            # 更新缓存
            self._cache[alias] = content
            self._mtime[alias] = current_mtime
            logger.debug(f"重新加载 prompt: {alias} from {prompt_path}")

            return content

        except Exception as e:
            logger.error(f"加载 prompt 失败: {alias} - {e}")
            return f"[加载失败: {e}]"

    def invalidate(self, alias: str):
        """使缓存失效"""
        if alias in self._cache:
            del self._cache[alias]
        if alias in self._mtime:
            del self._mtime[alias]
        logger.debug(f"使缓存失效: {alias}")

    def set_watch(self, enabled: bool):
        """
        设置文件监听（热重载）
        
        注意：这是简化实现，生产环境应使用 watchdog 库
        """
        if enabled and self._watcher is None:
            # 简化：使用简单的时间轮询
            self._watch_enabled = True
            self._last_poll = time.time()
            logger.info("启用 prompt 热重载（简单轮询模式）")
        elif not enabled:
            self._watch_enabled = False
            logger.info("禁用 prompt 热重载")

    def check_reload(self, alias: str, profile: AgentProfile) -> bool:
        """
        检查是否需要重新加载

        Returns:
            True 如果文件已修改需要重新加载

        Fix: TOCTOU 修复 - 使用 try/except 替代 exists()+stat() 分离检查，
        让 stat() 失败时自然返回 False（文件不存在→无需 reload）。
        """
        try:
            prompt_path = self.resolve_path(profile.prompt_path)
            current_mtime = prompt_path.stat().st_mtime
            old_mtime = self._mtime.get(alias, 0)

            if current_mtime > old_mtime:
                logger.info(f"检测到文件变化，重新加载: {alias}")
                return True
            return False
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return {
            "cached_aliases": list(self._cache.keys()),
            "cache_count": len(self._cache),
        }


# ==================== AgentSwitcher ====================

class AgentSwitcher:
    """管理当前活跃 Agent，支持角色切换"""

    def __init__(self, registry: AgentRegistry, loader: PromptLoader):
        self.registry = registry
        self.loader = loader
        self._active_alias: Optional[str] = None
        self._active_prompt: Optional[str] = None

    def switch_to(self, alias: str) -> dict:
        """
        切换到指定 Agent
        
        Returns:
            切换结果，包含 agent 信息和 prompt
        """
        profile = self.registry.get(alias)
        if not profile:
            return {
                "success": False,
                "error": f"Agent 不存在: {alias}"
            }
        
        # 检查是否需要重新加载 prompt
        needs_reload = self.loader.check_reload(alias, profile)
        
        # 加载 prompt
        prompt = self.loader.load(alias, profile)
        
        # 更新活跃状态
        old_alias = self._active_alias
        self._active_alias = alias
        self._active_prompt = prompt
        
        logger.info(f"切换 Agent: {old_alias} -> {alias} (role: {profile.role})")
        
        return {
            "success": True,
            "alias": alias,
            "name": profile.name,
            "role": profile.role,
            "description": profile.description,
            "prompt": prompt,
            "model_preference": profile.model_preference,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "reloaded": needs_reload
        }

    def get_active(self) -> dict:
        """获取当前活跃 Agent 信息"""
        if not self._active_alias:
            return {
                "active": False,
                "message": "没有活跃的 Agent"
            }
        
        profile = self.registry.get(self._active_alias)
        if not profile:
            return {
                "active": False,
                "error": "活跃 Agent 配置丢失"
            }
        
        return {
            "active": True,
            "alias": self._active_alias,
            "name": profile.name,
            "role": profile.role,
            "description": profile.description,
            "model_preference": profile.model_preference,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "prompt_length": len(self._active_prompt) if self._active_prompt else 0
        }

    def resolve_prompt(self, alias: str, **variables) -> str:
        """
        解析 prompt 模板，支持 {{variable}} 占位符
        
        Args:
            alias: Agent 别名
            **variables: 模板变量
        
        Returns:
            解析后的 prompt
        """
        profile = self.registry.get(alias)
        if not profile:
            return f"[Agent 不存在: {alias}]"
        
        # 加载 prompt
        prompt = self.loader.load(alias, profile)
        
        # 替换变量 {{variable}}
        def replace_var(match):
            var_name = match.group(1).strip()
            # 支持 context.key 语法
            if "." in var_name:
                parts = var_name.split(".", 1)
                if parts[0] == "context" and len(parts) == 2:
                    # 尝试从 variables 中获取 context
                    ctx = variables.get("context", {})
                    return str(ctx.get(parts[1], match.group(0)))
            return str(variables.get(var_name, match.group(0)))
        
        resolved = re.sub(r'\{\{([^}]+)\}\}', replace_var, prompt)
        
        return resolved


# ==================== ManagerSkill ====================

class ManagerSkill:
    """
    AgentManager 技能
    
    提供标准 Skill 接口：
    - query(capability, context): 查询能力
    - execute(action, params): 执行动作
    - notify(event, data): 接收事件通知
    """

    def __init__(
        self,
        agent_dir: Optional[Path] = None,
        map_file: Optional[Path] = None
    ):
        """
        初始化 ManagerSkill
        
        Args:
            agent_dir: Agent prompt 文件所在目录
            map_file: agent_map.json 文件路径
        """
        # 默认目录
        if agent_dir is None:
            agent_dir = Path.home() / ".agent-search" / "agents"
        self.agent_dir = Path(agent_dir)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.registry = AgentRegistry(map_file)
        self.loader = PromptLoader(self.agent_dir)
        self.switcher = AgentSwitcher(self.registry, self.loader)
        
        # 尝试加载已保存的注册表
        self.registry.load_from_file()

    def query(self, capability: str, context: Optional[dict] = None) -> dict:
        """
        查询技能能力
        
        Args:
            capability: 能力名称，如 "manager.register", "manager.switch"
            context: 上下文信息（未使用，保留接口兼容性）
        
        Returns:
            能力描述或执行结果
        """
        context = context or {}
        capability_map = {
            "manager.register": self._do_register,
            "manager.switch": self._do_switch,
            "manager.get_active": self._do_get_active,
            "manager.list": self._do_list,
            "manager.reload": self._do_reload,
            "manager.unregister": self._do_unregister,
            "manager.load_prompt": self._do_load_prompt,
        }
        
        if capability not in capability_map:
            return {
                "success": False,
                "error": {
                    "code": "CAPABILITY_NOT_FOUND",
                    "message": f"Capability {capability} not found",
                    "available": list(capability_map.keys())
                }
            }
        
        return capability_map[capability](context)

    def execute(self, action: str, params: dict) -> dict:
        """
        执行动作
        
        Args:
            action: 动作名称
            params: 参数
        
        Returns:
            执行结果
        """
        start_time = time.time()
        
        try:
            if action == "register":
                result = self._do_register(params)
            elif action == "switch":
                result = self._do_switch(params)
            elif action == "get_active":
                result = self._do_get_active(params)
            elif action == "list":
                result = self._do_list(params)
            elif action == "reload":
                result = self._do_reload(params)
            elif action == "unregister":
                result = self._do_unregister(params)
            elif action == "load_prompt":
                result = self._do_load_prompt(params)
            elif action == "save":
                result = self.registry.save_to_file()
            elif action == "resolve_prompt":
                result = self._do_resolve_prompt(params)
            else:
                return {
                    "success": False,
                    "error": {
                        "code": "ACTION_NOT_FOUND",
                        "message": f"Action {action} not found"
                    }
                }
            
            # 包装结果
            return {
                "success": result.get("success", True),
                "data": result,
                "meta": {
                    "skill": "manager",
                    "action": action,
                    "duration_ms": int((time.time() - start_time) * 1000)
                }
            }
            
        except Exception as e:
            logger.error(f"执行失败: {action} - {e}")
            return {
                "success": False,
                "error": {
                    "code": "EXECUTION_ERROR",
                    "message": str(e)
                }
            }

    def notify(self, event: str, data: dict):
        """
        接收事件通知
        """
        logger.debug(f"收到事件: {event}, 数据: {data}")
        
        if event == "file_changed":
            # 处理文件变化事件，触发热重载
            alias = data.get("alias")
            if alias:
                self.loader.invalidate(alias)
                logger.info(f"热重载触发: {alias}")
        
        elif event == "agent_updated":
            # Agent 配置更新
            alias = data.get("alias")
            if alias:
                self.loader.invalidate(alias)

    # ==================== 内部实现 ====================

    def _do_register(self, params: dict) -> dict:
        """注册 Agent"""
        alias = params.get("alias")
        if not alias:
            return {"success": False, "error": "alias 不能为空"}
        
        # 构建 profile
        profile = AgentProfile(
            alias=alias,
            name=params.get("name", alias),
            role=params.get("role", "assistant"),
            description=params.get("description", ""),
            prompt_path=params.get("prompt_path", f"{alias}.txt"),
            model_preference=params.get("model_preference"),
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 4096)
        )
        
        # 注册
        result = self.registry.register(alias, profile)
        
        # 尝试保存
        if result.get("success"):
            self.registry.save_to_file()
        
        return result

    def _do_unregister(self, params: dict) -> dict:
        """注销 Agent"""
        alias = params.get("alias")
        if not alias:
            return {"success": False, "error": "alias 不能为空"}
        
        result = self.registry.unregister(alias)
        
        # 尝试保存
        if result.get("success"):
            self.registry.save_to_file()
        
        return result

    def _do_switch(self, params: dict) -> dict:
        """切换 Agent"""
        alias = params.get("alias")
        if not alias:
            return {"success": False, "error": "alias 不能为空"}
        
        return self.switcher.switch_to(alias)

    def _do_get_active(self, params: dict) -> dict:
        """获取当前活跃 Agent"""
        return self.switcher.get_active()

    def _do_list(self, params: dict) -> dict:
        """列出所有 Agent"""
        return {
            "success": True,
            "agents": self.registry.list_all(),
            "count": len(self.registry._agents)
        }

    def _do_reload(self, params: dict) -> dict:
        """重新加载 Agent 注册表"""
        self.registry.load_from_file()
        return {
            "success": True,
            "message": "重新加载完成",
            "agents": self.registry.list_all()
        }

    def _do_load_prompt(self, params: dict) -> dict:
        """加载指定 Agent 的 prompt"""
        alias = params.get("alias")
        if not alias:
            return {"success": False, "error": "alias 不能为空"}
        
        profile = self.registry.get(alias)
        if not profile:
            return {"success": False, "error": f"Agent 不存在: {alias}"}
        
        prompt = self.loader.load(alias, profile)
        return {
            "success": True,
            "alias": alias,
            "prompt": prompt,
            "length": len(prompt)
        }

    def _do_resolve_prompt(self, params: dict) -> dict:
        """解析 prompt 模板"""
        alias = params.get("alias")
        if not alias:
            return {"success": False, "error": "alias 不能为空"}
        
        # 提取变量（排除 alias）
        variables = {k: v for k, v in params.items() if k != "alias"}
        
        resolved = self.switcher.resolve_prompt(alias, **variables)
        return {
            "success": True,
            "alias": alias,
            "resolved": resolved,
            "length": len(resolved)
        }


# ==================== 便捷函数 ====================

def create_manager_skill(
    agent_dir: Optional[str] = None,
    map_file: Optional[str] = None
) -> ManagerSkill:
    """
    创建 ManagerSkill 实例的便捷函数
    
    Args:
        agent_dir: Agent prompt 文件目录
        map_file: agent_map.json 路径
    
    Returns:
        ManagerSkill 实例
    """
    return ManagerSkill(
        agent_dir=Path(agent_dir) if agent_dir else None,
        map_file=Path(map_file) if map_file else None
    )
