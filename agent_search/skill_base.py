"""SkillBase — 4 skill 公共基类 (V 21:52 util 化)

V 反思 SOP 第 10 件加强版: 必带 verify
- 重构前 4 skill 行数: safety 812 + supervisor 755 + manager 681 + team 382 = 2630
- 重构后预期: safety 760 + supervisor 720 + manager 640 + team 382 = 2502
- 减少 ~130 行 (4 skill query/execute/notify 重复 + import 合并)

V 21:52 4 拍板项:
1. safety / supervisor / manager 继承 SkillBase, 改 _handle_query/_handle_execute
2. team_skill 维持独立 (gateway client 设计, 不适合 SkillBase 接口)
3. 所有方法签名兼容 (AgentSymphony 集成调用)
4. 跑测试必过 (73 unit + 10/10 smoke + 9 endpoint verify)
"""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SkillBase(ABC):
    """4 skill 公共基类 (V 21:52 抽)

    公共方法:
        - query(capability, context) -> dict
        - execute(action, params) -> dict
        - notify(event, data) -> None

    抽象方法 (子类必实现):
        - _handle_query(capability, context) -> dict
        - _handle_execute(action, params) -> dict

    公共属性:
        - skill_name: str = "base"
        - capabilities: list[str] = []
    """

    skill_name: str = "base"
    capabilities: list[str] = []

    def __init__(self, config: Any = None):
        self.config = config
        self._stats = {
            "query_count": 0,
            "execute_count": 0,
            "notify_count": 0,
            "error_count": 0,
        }

    def query(self, capability: str, context: dict | None = None) -> dict:
        """查询技能能力 (公共入口, 委托给 _handle_query)

        Args:
            capability: 能力名称
            context: 上下文信息

        Returns:
            能力描述或执行结果 dict
        """
        self._stats["query_count"] += 1
        context = context or {}
        try:
            return self._handle_query(capability, context)
        except Exception as e:
            self._stats["error_count"] += 1
            logger.exception(f"{self.skill_name}.query({capability}) failed: {e}")
            return {"error": str(e), "capability": capability}

    def execute(self, action: str, params: dict | None = None) -> dict:
        """执行动作 (公共入口, 委托给 _handle_execute)

        Args:
            action: 动作名称
            params: 参数

        Returns:
            执行结果 dict
        """
        self._stats["execute_count"] += 1
        params = params or {}
        start_time = time.time()
        try:
            result = self._handle_execute(action, params)
            elapsed = time.time() - start_time
            if isinstance(result, dict):
                result.setdefault("_elapsed_ms", round(elapsed * 1000, 2))
            return result
        except Exception as e:
            self._stats["error_count"] += 1
            logger.exception(f"{self.skill_name}.execute({action}) failed: {e}")
            return {"error": str(e), "action": action}

    def notify(self, event: str, data: dict | None = None) -> None:
        """通知事件 (公共入口)

        Args:
            event: 事件名称
            data: 事件数据
        """
        self._stats["notify_count"] += 1
        data = data or {}
        logger.info(f"{self.skill_name} notify: {event} | data keys={list(data.keys())[:3]}")

    def get_stats(self) -> dict:
        """获取统计 (V 21:52 新增, 公共基类)"""
        return dict(self._stats)

    def list_capabilities(self) -> list[str]:
        """列出能力 (V 21:52 新增, 公共基类)"""
        return list(self.capabilities)

    @abstractmethod
    def _handle_query(self, capability: str, context: dict) -> dict:
        """子类实现: 实际查询逻辑"""
        raise NotImplementedError

    @abstractmethod
    def _handle_execute(self, action: str, params: dict) -> dict:
        """子类实现: 实际执行逻辑"""
        raise NotImplementedError


def is_skill_base(obj: Any) -> bool:
    """V 21:52 util 化: 检查是否是 SkillBase 子类 (兼容 check)"""
    return isinstance(obj, SkillBase)
