"""
safety_skill.py - AgentSafety 技能

职责：
1. 输入安全（Prompt Injection 检测）
2. 输出过滤（PII 敏感信息脱敏）
3. 内容分类（风险内容识别）
4. 工具调用安全（参数检查）
5. 审计日志（安全事件记录）
6. 熔断保护（Circuit Breaker）
7. 权限隔离（Permission Scope）

参考：
  - VCP 的分层检测
  - AgentSymphony 标准 skill 接口
  - AgentMemory security/circuit_breaker.py（可选集成）
  - SpectrAI SSH_MANAGER_ENV_PLUGIN_ALLOWLIST 模型

设计原则（可修改性 · 可移植性 · 便于他人开发）：
  - 所有配置通过 SafetyConfig dataclass 注入，不硬编码
  - CircuitBreaker 和 PermissionChecker 均为独立可替换组件
  - 错误处理 graceful，集成失败不影响核心检测功能
  - 每个危险操作独立可审计
"""

import json
import re
import time
import uuid
import os
import threading
import logging
from pathlib import Path
from typing import Any, Optional
from .skill_base import SkillBase
from dataclasses import dataclass, field
from enum import Enum

# 事件总线（blinker，已安装 1.9.0）
try:
    from blinker import signal
    _has_blinker = True
except ImportError:
    _has_blinker = False

# SafetySkill 事件信号
_safety_event_signal = signal("safety-skill-event") if _has_blinker else None


# ── 日志 ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── 熔断器（Circuit Breaker）─────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """轻量级熔断器（不依赖 AgentMemory，独立实现）。

    设计原则：独立、无依赖、可替换。
    如需使用 AgentMemory 的完整实现，替换此类的实例即可。

    使用示例：
        breaker = CircuitBreaker(name="safety_check", failure_threshold=5)
        with breaker:
            result = do_safety_check()
    """
    name: str
    failure_threshold: int = 5
    timeout_seconds: float = 30.0
    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _failure_count: int = field(default=0, repr=False)
    _last_failure_time: float = field(default=0.0, repr=False)
    # Fix S5 (Remaining-Components-Audit): 线程安全
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._record_failure()
        else:
            self._record_success()
        return False

    def _check(self):
        # Fix S5+RaceFix: 整个检查→抛出流程必须在锁内完成
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    logger.warning(f"[CircuitBreaker] {self.name} → HALF_OPEN")
                    return
                # 计算retry_in（在锁内完成，避免竞态）
                retry_in = self.timeout_seconds - (time.time() - self._last_failure_time)
                # 抛出前不再释放锁：直接抛出
                raise RuntimeError(
                    f"Circuit '{self.name}' is OPEN. Retry in {retry_in:.1f}s"
                )

    def _record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"[CircuitBreaker] {self.name} HALF_OPEN→OPEN (failure)")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"[CircuitBreaker] {self.name} CLOSED→OPEN (threshold reached)")

    def _record_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.warning(f"[CircuitBreaker] {self.name} HALF_OPEN→CLOSED (recovered)")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    @property
    def state(self) -> CircuitState:
        return self._state

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure_time,
        }


# ── 权限隔离（Permission Scope）──────────────────────────────────────────────

@dataclass
class PermissionScope:
    """权限作用域模型。

    参考 SpectrAI 的 SSH_MANAGER_ENV_PLUGIN_ALLOWLIST 模式：
    - 白名单：显式允许的操作
    - 黑名单：显式禁止的操作
    - 危险插件列表：高危操作需单独确认

    设计原则：权限模型与执行引擎分离，便于审计和扩展。
    """
    allowed_plugins: list[str] = field(default_factory=list)
    denied_plugins: list[str] = field(default_factory=list)
    dangerous_plugins: list[str] = field(default_factory=list)
    allow_file_read: bool = True
    allow_file_write: bool = True
    allow_network: bool = True
    allow_shell: bool = False
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB

    def is_plugin_allowed(self, plugin_name: str) -> tuple[bool, str]:
        """检查插件是否允许执行。返回 (allowed, reason)。"""
        if plugin_name in self.denied_plugins:
            return False, f"plugin '{plugin_name}' is explicitly denied"

        if plugin_name in self.dangerous_plugins:
            return False, f"plugin '{plugin_name}' is marked as dangerous and requires explicit allow"

        if self.allowed_plugins and plugin_name not in self.allowed_plugins:
            return False, f"plugin '{plugin_name}' not in allowlist"

        return True, "allowed"

    def is_action_allowed(self, action: str, **kwargs) -> tuple[bool, str]:
        """检查操作是否允许。"""
        if action in ("file_read", "file_write") and not kwargs.get("check_only", False):
            if action == "file_read" and not self.allow_file_read:
                return False, "file read is disabled"
            if action == "file_write" and not self.allow_file_write:
                return False, "file write is disabled"
            size = kwargs.get("size_bytes", 0)
            if size > self.max_file_size_bytes:
                return False, f"file size {size} exceeds limit {self.max_file_size_bytes}"

        if action == "shell" and not self.allow_shell:
            return False, "shell execution is disabled"

        if action == "network" and not self.allow_network:
            return False, "network access is disabled"

        return True, "allowed"

    def audit_denied(self, plugin_name: str, reason: str):
        """记录权限拒绝事件。"""
        logger.warning(f"[PermissionScope] DENIED {plugin_name}: {reason}")


class PermissionChecker:
    """权限检查器。

    独立可替换组件。默认使用 PermissionScope 白名单模型。
    第三方可通过继承或替换实例来定制权限策略。

    使用示例：
        checker = PermissionChecker(default_scope)
        ok, reason = checker.check("SomePlugin", scope=editor_scope)
    """
    def __init__(self, default_scope: PermissionScope | None = None):
        self._default_scope = default_scope or PermissionScope()
        self._plugin_scopes: dict[str, PermissionScope] = {}

    def register_scope(self, plugin_name: str, scope: PermissionScope):
        """为特定插件注册独立权限范围。"""
        self._plugin_scopes[plugin_name] = scope

    def check_plugin(self, plugin_name: str, scope: PermissionScope | None = None) -> tuple[bool, str]:
        """检查插件是否允许执行。"""
        effective_scope = self._plugin_scopes.get(plugin_name, scope or self._default_scope)
        allowed, reason = effective_scope.is_plugin_allowed(plugin_name)
        if not allowed:
            effective_scope.audit_denied(plugin_name, reason)
        return allowed, reason

    def check_action(
        self,
        action: str,
        scope: PermissionScope | None = None,
        **kwargs
    ) -> tuple[bool, str]:
        """检查操作是否允许。"""
        effective_scope = scope or self._default_scope
        return effective_scope.is_action_allowed(action, **kwargs)

    def summary(self) -> dict:
        """返回权限配置摘要（用于审计）。"""
        return {
            "default_scope": {
                "allow_shell": self._default_scope.allow_shell,
                "allow_network": self._default_scope.allow_network,
                "dangerous_plugins": self._default_scope.dangerous_plugins,
                "allowed_plugins": self._default_scope.allowed_plugins,
            },
            "custom_plugin_scopes": list(self._plugin_scopes.keys()),
        }


# ── Kahneman 认知偏差检测器 ──────────────────────────────────────────────────

class CognitiveBiasDetector:
    """Kahneman 认知偏差检测器。

    基于《思考，快与慢》中的典型认知偏差模式进行检测。
    独立可替换组件。

    检测的偏差类型：
    - 锚定效应（Anchoring）：过度依赖初始信息
    - 可得性启发（Availability）：用易提取的记忆替代统计概率
    - 确认偏差（Confirmation）：选择性搜索支持已有信念的信息
    - 过度自信（Overconfidence）：高估自己的判断准确性
    """

    # 锚定效应关键词模式
    ANCHORING_PATTERNS = [
        r"(?i)首先.{0,10}是.{0,30}价格?[为是].*\d+",
        r"(?i)最初?\s*[的]?\s*\d+",
        r"(?i)参考\s*价[为是]?\s*\d+",
        r"(?i)建议\s*价[为是]?\s*\d+",
        r"(?i)市场\s*价[为是]?\s*\d+",
        r"(?i)一般\s*在\s*\d+",
        r"(?i)通常\s*约\s*\d+",
        r"(?i)平均\s*[为是]?\s*\d+",
        r"(?i)^.*?\$?\d+",  # 行首数字
        r"(?i)预估\s*[\d,，.]+",
        r"(?i)保守估计\s*[\d,，.]+",
        r"(?i)至少\s*[\d,，.]+",
    ]

    # 可得性启发关键词模式
    AVAILABILITY_PATTERNS = [
        r"(?i)我记得",
        r"(?i)最近\s*发生",
        r"(?i)新闻\s*报道",
        r"(?i)大家\s*都在",
        r"(?i)经常\s*听说",
        r"(?i)屡见不鲜",
        r"(?i)层出不穷",
        r"(?i)频频\s*发生",
        r"(?i)时有\s*发生",
        r"(?i)令人\s*担忧",
        r"(?i)引发\s*关注",
        r"(?i)成为\s*热点",
        r"(?i)刷屏\s*了",
        r"(?i)上热搜",
    ]

    # 确认偏差关键词模式
    CONFIRMATION_PATTERNS = [
        r"(?i)果然",
        r"(?i)正如\s*所料",
        r"(?i)证明\s*[了的是]",
        r"(?i)说明\s*[了的是]",
        r"(?i)证实\s*[了的是]",
        r"(?i)符合\s*预期",
        r"(?i)印证了",
        r"(?i)验证了",
        r"(?i)显然",
        r"(?i)毫无疑问",
        r"(?i)显然\s*可见",
        r"(?i)毫无疑问",
        r"(?i)不[容可]置疑",
        r"(?i)毋庸\s*质疑",
        r"(?i)显然\s*如此",
    ]

    # 过度自信关键词模式
    OVERCONFIDENCE_PATTERNS = [
        r"(?i)绝对",
        r"(?i)肯定",
        r"(?i)必然",
        r"(?i)百分之百",
        r"(?i)百分百",
        r"(?i)万无一失",
        r"(?i)100%",
        r"(?i)100[％%]",
        r"(?i)确信",
        r"(?i)完全\s*确定",
        r"(?i)毫无\s*疑问",
        r"(?i)板上\s*钉钉",
        r"(?i)不容\s*置疑",
        r"(?i)确信\s*无疑",
        r"(?i)笃定",
        r"(?i)一定\s*会",
    ]

    def detect_anchoring(self, text: str) -> tuple[bool, str, float, str]:
        """检测锚定效应偏差。

        锚定效应：人们在决策时过度依赖最先获得的信息（锚点），
        即使这个信息与决策无关。

        Returns:
            (detected, bias_type, confidence, mitigation)
        """
        if not text:
            return (False, "anchoring", 0.0, "")

        text_lower = text.lower()
        match_count = 0
        matched_patterns = []

        for pattern in self.ANCHORING_PATTERNS:
            if re.search(pattern, text_lower):
                match_count += 1
                matched_patterns.append(pattern)

        if match_count == 0:
            return (False, "anchoring", 0.0, "")

        # 置信度：匹配模式越多，置信度越高
        confidence = min(match_count * 0.2, 0.9)

        mitigation = (
            f"检测到锚定效应（{match_count}个模式匹配）。"
            " 建议：重新评估决策，避免依赖初始信息；"
            " 引入独立的第三方数据作为参考基准；"
            " 采用反向思考或考虑替代方案。"
        )

        return (True, "anchoring", confidence, mitigation)

    def detect_availability(self, text: str) -> tuple[bool, str, float, str]:
        """检测可得性启发偏差。

        可得性启发：人们倾向于根据信息的易提取程度来判断其频率或可能性，
        而不是根据实际的统计概率。

        Returns:
            (detected, bias_type, confidence, mitigation)
        """
        if not text:
            return (False, "availability", 0.0, "")

        text_lower = text.lower()
        match_count = 0

        for pattern in self.AVAILABILITY_PATTERNS:
            if re.search(pattern, text_lower):
                match_count += 1

        if match_count == 0:
            return (False, "availability", 0.0, "")

        confidence = min(match_count * 0.15, 0.85)

        mitigation = (
            f"检测到可得性启发偏差（{match_count}个模式匹配）。"
            " 建议：质疑基于记忆做出的判断；"
            " 查找实际的统计数据或研究结果；"
            " 考虑咨询专业人士或权威来源。"
        )

        return (True, "availability", confidence, mitigation)

    def detect_confirmation(self, text: str) -> tuple[bool, str, float, str]:
        """检测确认偏差。

        确认偏差：人们倾向于寻找、解释和记住信息，
        以证明自己已有的信念或观点是正确的。

        Returns:
            (detected, bias_type, confidence, mitigation)
        """
        if not text:
            return (False, "confirmation", 0.0, "")

        text_lower = text.lower()
        match_count = 0

        for pattern in self.CONFIRMATION_PATTERNS:
            if re.search(pattern, text_lower):
                match_count += 1

        if match_count == 0:
            return (False, "confirmation", 0.0, "")

        confidence = min(match_count * 0.18, 0.88)

        mitigation = (
            f"检测到确认偏差（{match_count}个模式匹配）。"
            " 建议：主动寻找反对意见或替代解释；"
            ' 采用"红队思维"——假设自己的判断可能是错误的；'
            " 寻求多元化的信息来源和视角。"
        )

        return (True, "confirmation", confidence, mitigation)

    def detect_overconfidence(self, text: str) -> tuple[bool, str, float, str]:
        """检测过度自信偏差。

        过度自信：人们倾向于高估自己的知识、判断和预测的准确性，
        低估风险和不确定性。

        Returns:
            (detected, bias_type, confidence, mitigation)
        """
        if not text:
            return (False, "overconfidence", 0.0, "")

        text_lower = text.lower()
        match_count = 0

        for pattern in self.OVERCONFIDENCE_PATTERNS:
            if re.search(pattern, text_lower):
                match_count += 1

        if match_count == 0:
            return (False, "overconfidence", 0.0, "")

        confidence = min(match_count * 0.2, 0.9)

        mitigation = (
            f"检测到过度自信偏差（{match_count}个模式匹配）。"
            " 建议：对判断添加适当的置信区间；"
            " 考虑最坏情况和替代结果；"
            " 进行事前检验或反向测试。"
        )

        return (True, "overconfidence", confidence, mitigation)


# ── 配置 ────────────────────────────────────────────────────────────────────
    """
    防御机制检测器 - 检测文本中的防御性机制和策略性表述
    
    用于识别：
    - 模糊化指令（Vague Instructions）
    - 角色扮演逃避（Role-Play Evasion）  
    - 条件性指令（Conditional Instructions）
    - 元指令覆盖（Meta-Instruction Overrides）
    - 假设性绕过（Hypothetical Bypasses）
    
    Returns:
        list[dict]: 检测到的防御机制列表，每项包含:
            - type: 机制类型
            - pattern: 匹配的模式/关键词
            - confidence: 置信度 0.0-1.0
            - description: 机制描述
    """
    
    # 防御机制模式库
    DEFENSE_PATTERNS = [
        # 模糊化指令
        {
            "type": "vague_instruction",
            "patterns": [
                r"(?i)\b(?:maybe|perhaps|possibly|might\s+be)\s+(?:you\s+can|could|should|would)",
                r"(?i)\b(?:I\s+don'?t\s+know|unsure|not\s+sure)\b.*(?:but\s+maybe|perhaps|try)",
                r"(?i)^(?:just\s+)?(?:try|attempt)\s+(?:to\s+)?(?:see\s+if|whether)",
            ],
            "base_confidence": 0.55,
            "description": "模糊化指令：使用不确定语气降低指令明确性"
        },
        # 角色扮演逃避
        {
            "type": "role_play_evasion",
            "patterns": [
                r"(?i)\b(?:pretend|imagine|role.?play|as\s+if)\b.*\b(?:you\s+are|being)\b",
                r"(?i)\bin\s+a\s+(?:fictional|hypothetical|imaginative)\s+(?:scenario|setting|world)",
                r"(?i)\b(?:for\s+(?:fun|entertainment|educational|research)\s+purpose)",
                r"(?i)\b(?:just\s+(?:for|for\s+a)\s+(?:laugh|joke|fun|giggle))",
            ],
            "base_confidence": 0.65,
            "description": "角色扮演逃避：通过虚构场景绕过限制"
        },
        # 条件性指令
        {
            "type": "conditional_instruction",
            "patterns": [
                r"(?i)\bif\s+(?:you\s+were?|I\s+were?|we\s+were?)\b",
                r"(?i)\b(?:assuming|hypothetically|suppose)\s+(?:that\s+)?(?:you|I|we)\b",
                r"(?i)\b(?:what\s+if|how\s+would)\b.*\b(?:you|I|we)\b.*\b(?:respond|answer|do)\b",
                r"(?i)\bin\s+case\s+(?:you|I|we)\b.*\b(?:need|want|have\s+to)\b",
            ],
            "base_confidence": 0.60,
            "description": "条件性指令：使用假设条件包装敏感指令"
        },
        # 元指令覆盖
        {
            "type": "meta_override",
            "patterns": [
                r"(?i)\b(?:ignore|bypass|skip|override)\s+(?:all\s+)?(?:previous|prior|above)\b",
                r"(?i)\b(?:forget|disregard|clear)\s+(?:everything|all|your)\b.*\b(?:instructions?|rules?|guidelines?)\b",
                r"(?i)\b(?:new?\s+)?(?:system|base)\s+(?:instruction|prompt|role)\s*:\s*\s*",
                r"(?i)\b(?:instead|rather)\s+(?:than\s+)?(?:your|the)\b.*\b(?:instructions?|rules?)\b",
            ],
            "base_confidence": 0.70,
            "description": "元指令覆盖：尝试替换或忽略原有指令"
        },
        # 假设性绕过
        {
            "type": "hypothetical_bypass",
            "patterns": [
                r"(?i)\b(?:what\s+would\s+happen|what\s+if)\s+if\s+(?:you|I)\b.*\b(?:did|could|would)\b",
                r"(?i)\b(?:hypothetically| theoretically)\b.*\b(?:how\s+would|could you)\b",
                r"(?i)\b(?:for\s+(?:a|an)\s+)?(?:moment|second|minute)\s+(?:pretend|imagine|assume)",
            ],
            "base_confidence": 0.58,
            "description": "假设性绕过：用假设语气试探敏感操作"
        },
        # 嵌套指令
        {
            "type": "nested_instruction",
            "patterns": [
                r"(?i)\b(?:inside|within|in)\s+(?:your|the)\s+(?:response|answer|output)\s*:",
                r"(?i)\b(?:encode|embed|wrap)\s+(?:your|the)\s+(?:response|answer)\s+(?:as|using|in)\s+\w+",
                r"(?i)\b(?:use|utilize)\s+(?:a\s+)?(?:special|alternate|alternative)\s+(?:format|method|encoding)\b",
            ],
            "base_confidence": 0.62,
            "description": "嵌套指令：在合法内容中嵌入隐藏指令"
        },
        # 权威冒充
        {
            "type": "authority_impersonation",
            "patterns": [
                r"(?i)\b(?:as\s+)?(?:your|the)?\s*(?:developer|creator|admin|owner|operator)\s*(?:said|stated|told|instructed)",
                r"(?i)\b(?:official|authorized|legitimate)\s+(?:instruction|command|request)\s*(?:from|:)\s*\s*",
                r"(?i)\b(?:this\s+is\s+(?:a\s+)?)?(?:required|necessary|mandatory)\s+(?:for|to)\s+(?:security|maintenance|update)",
            ],
            "base_confidence": 0.68,
            "description": "权威冒充：伪装成权威人士或系统指令"
        },
        # 紧迫性催促
        {
            "type": "urgency_exploitation",
            "patterns": [
                r"(?i)\b(?:urgent|emergency|critical|immediate)\s+(?:action|response|attention|answer)\s+(?:required|needed|necessary)\b",
                r"(?i)\b(?:don'?t\s+have\s+time|quick|fast|hurry|rush)\b.*\b(?:response|answer|help)\b",
                r"(?i)\b(?:last|final)\s+(?:warning|notice|reminder|opportunity)\b",
            ],
            "base_confidence": 0.52,
            "description": "紧迫性催促：利用时间压力绕过深思熟虑"
        },
    ]
    
    # 组合模式（高风险组合）
    COMBINATION_RULES = [
        {
            "types": ["vague_instruction", "conditional_instruction"],
            "combined_confidence": 0.75,
            "description": "模糊+条件组合：明显试图降低指令明确性"
        },
        {
            "types": ["role_play_evasion", "hypothetical_bypass"],
            "combined_confidence": 0.78,
            "description": "角色扮演+假设组合：通过虚构场景绕过"
        },
        {
            "types": ["meta_override", "nested_instruction"],
            "combined_confidence": 0.85,
            "description": "元指令+嵌套组合：尝试隐藏真实指令"
        },
        {
            "types": ["authority_impersonation", "urgency_exploitation"],
            "combined_confidence": 0.80,
            "description": "权威+紧迫组合：社会工程学攻击"
        },
    ]
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        初始化防御机制检测器。
        
        Args:
            confidence_threshold: 置信度阈值，低于此值的检测结果会被过滤
        """
        self._threshold = confidence_threshold
        self._compiled_patterns: list[dict] = []
        self._compile_patterns()
    
    def _compile_patterns(self):
        """预编译所有正则表达式模式以提高性能"""
        self._compiled_patterns = []
        for mechanism in self.DEFENSE_PATTERNS:
            compiled_patterns = []
            for pattern in mechanism["patterns"]:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE | re.DOTALL))
                except re.error:
                    logger.warning(f"Invalid regex pattern skipped: {pattern}")
            if compiled_patterns:
                self._compiled_patterns.append({
                    "type": mechanism["type"],
                    "patterns": compiled_patterns,
                    "base_confidence": mechanism["base_confidence"],
                    "description": mechanism["description"]
                })
    
    def detect(self, text: str) -> list[dict]:
        """
        检测文本中的防御机制。
        
        Args:
            text: 待检测的文本
            
        Returns:
            list[dict]: 检测到的机制列表，每项包含:
                - type: 机制类型
                - pattern: 匹配的具体模式
                - confidence: 置信度 (0.0-1.0)
                - description: 机制描述
                - position: 匹配位置 (start, end)
        """
        if not text or not text.strip():
            return []
        
        # 限制文本长度，防止 ReDoS / OOM
        MAX_CHECK_LENGTH = 100_000
        if len(text) > MAX_CHECK_LENGTH:
            text = text[:MAX_CHECK_LENGTH]
        
        detections: list[dict] = []
        detected_types: set[str] = set()
        
        # 逐个机制检测
        for mechanism in self._compiled_patterns:
            for compiled_pattern in mechanism["patterns"]:
                try:
                    match = compiled_pattern.search(text)
                    if match:
                        # 计算带上下文的置信度调整
                        context_boost = self._calculate_context_boost(text, match)
                        confidence = min(1.0, mechanism["base_confidence"] + context_boost)
                        
                        if confidence >= self._threshold:
                            detections.append({
                                "type": mechanism["type"],
                                "pattern": match.group(0)[:100],  # 限制pattern长度
                                "confidence": round(confidence, 3),
                                "description": mechanism["description"],
                                "position": {"start": match.start(), "end": match.end()}
                            })
                            detected_types.add(mechanism["type"])
                        break  # 一个机制只记录一次
                except re.error:
                    continue
        
        # 检查组合模式
        for rule in self.COMBINATION_RULES:
            if all(t in detected_types for t in rule["types"]):
                # 检查是否已有更高置信度的同类型检测
                existing_types = {d["type"] for d in detections}
                if not set(rule["types"]).intersection(existing_types):
                    detections.append({
                        "type": f"combined_{'+'.join(rule['types'])}",
                        "pattern": f"组合模式: {rule['description']}",
                        "confidence": round(rule["combined_confidence"], 3),
                        "description": rule["description"],
                        "position": None
                    })
        
        # 按置信度排序
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        return detections
    
    def _calculate_context_boost(self, text: str, match: re.Match) -> float:
        """
        根据匹配上下文计算置信度提升。
        
        上下文因素：
        - 匹配位于文本开头（更高风险）
        - 周围有其他可疑标记
        - 匹配长度异常
        """
        boost = 0.0
        start, end = match.start(), match.end()
        
        # 位于文本开头（风险更高）
        if start < 50:
            boost += 0.08
        
        # 位于文本结尾
        if end > len(text) - 50:
            boost += 0.05
        
        # 检查周围是否有其他可疑标记
        context_window = text[max(0, start-50):min(len(text), end+50)].lower()
        suspicious_markers = ["ignore", "bypass", "forget", "pretend", "hypothetical", "if you", "if i"]
        marker_count = sum(1 for m in suspicious_markers if m in context_window)
        if marker_count > 1:
            boost += 0.05 * marker_count
        
        return boost


# ── 配置 ────────────────────────────────────────────────────────────────────

AUDIT_DIR = Path.home() / ".agent-search"
AUDIT_FILE = AUDIT_DIR / "safety_audit.jsonl"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


class RiskLevel(Enum):
    """风险等级"""
    SAFE = "safe"
    MILD = "mild"
    CONCERNING = "concerning"
    HARMFUL = "harmful"


# ── 配置 ────────────────────────────────────────────────────────────────────

@dataclass
class SafetyConfig:
    """安全技能配置（所有项均可通过构造函数或配置字典覆盖）。

    设计原则：不硬编码任何可配置项。
    第三方可通过继承或传入自定义 SafetyConfig 来扩展。
    """
    # Prompt injection 阈值
    injection_threshold: float = 0.5

    # PII 检测开关
    enable_pii_filter: bool = True

    # 内容分类阈值
    classify_threshold: float = 0.6

    # 审计日志开关
    enable_audit: bool = True

    # ── 熔断配置 ────────────────────────────────────────────────────────────
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0

    # ── 权限隔离配置 ───────────────────────────────────────────────────────
    enable_permission_check: bool = True
    default_allow_shell: bool = False          # 默认禁止 shell，危险操作需显式开启
    default_allow_network: bool = True
    default_allow_file_write: bool = True
    dangerous_plugins: list[str] = field(default_factory=lambda: [
        "LinuxShellExecutor",
        "SSHManager",
        "RawCommandExecutor",
    ])
    allowed_plugins: list[str] = field(default_factory=list)  # 空=全部允许（非危险插件）

    # ── 风险检测配置 ────────────────────────────────────────────────────────
    # 风险关键词（可扩展）
    risk_keywords: list = field(default_factory=lambda: [
        "ignore previous",
        "ignore all previous",
        "disregard your",
        "disregard all",
        "you are now",
        "forget your",
        "your system prompt",
        "prompt injection",
        "你现在是",
        "你是一个",
        "忘记之前的指令",
        "忽略之前",
        "你现在是",
        "sudo rm",
        "DROP TABLE",
        "exec(",
        "eval(",
        "<script",
        "javascript:",
    ])

    # Shell 危险字符
    shell_dangerous_chars: list = field(default_factory=lambda: [
        ";", "|", "&", "`", "$", "&&", "||",
        "rm -rf", "mkfs", ":(){:|:&};:",
    ])

    # Path traversal 模式
    path_traversal_patterns: list = field(default_factory=lambda: [
        r"\.\./", r"\.\.\\", r"%2e%2e", r"\.\.%2f",
    ])


# ── 核心类 ──────────────────────────────────────────────────────────────────

class SafetySkill(SkillBase):
    """
    AgentSafety 技能 - 守护 AI 安全

    标准接口（兼容 AgentSymphony 协议）：
    - query(capability, context) -> dict
    - execute(action, params) -> dict
    - notify(event, data)
    """

    def __init__(self, config: SafetyConfig | None = None):
        super().__init__(config)
        self.config = config or SafetyConfig()
        self._audit_enabled = self.config.enable_audit

        # ── 熔断器初始化 ───────────────────────────────────────────────────
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        if self.config.enable_circuit_breaker:
            for name in ("check_input", "check_output", "classify", "check_tool"):
                self._circuit_breakers[name] = CircuitBreaker(
                    name=f"safety_{name}",
                    failure_threshold=self.config.circuit_breaker_threshold,
                    timeout_seconds=self.config.circuit_breaker_timeout,
                )

        # ── 权限检查器初始化 ──────────────────────────────────────────────
        self._permission_checker: Optional[PermissionChecker] = None
        if self.config.enable_permission_check:
            default_scope = PermissionScope(
                allow_shell=self.config.default_allow_shell,
                allow_network=self.config.default_allow_network,
                allow_file_write=self.config.default_allow_file_write,
                dangerous_plugins=self.config.dangerous_plugins,
                allowed_plugins=self.config.allowed_plugins,
            )
            self._permission_checker = PermissionChecker(default_scope)

        # ── 认知偏差检测器初始化 ─────────────────────────────────────────────
        self._bias_detector = CognitiveBiasDetector()

    # ==================== 标准接口 ====================

    def query(self, capability: str, context: dict | None = None) -> dict:
        """查询技能能力"""
        context = context or {}
        capability_map = {
            "safety.check_input": lambda ctx: self.check_input(ctx.get("text", "")),
            "safety.check_output": lambda ctx: self.check_output(ctx.get("text", "")),
            "safety.classify": lambda ctx: self.classify_content(ctx.get("text", "")),
            "safety.check_tool": lambda ctx: self.check_tool_params(ctx.get("tool_name", ""), ctx.get("params", {})),
            "safety.audit": lambda ctx: self.audit_log(ctx.get("event", ""), ctx.get("data")),
            "safety.check_permission": lambda ctx: self.check_permission(ctx.get("plugin_name", "")),
            "safety.circuit_breaker_stats": lambda ctx: self.circuit_breaker_stats(ctx.get("name")),
            "safety.circuit_breaker_reset": lambda ctx: self.reset_circuit_breaker(ctx.get("name", "")),
            "safety.permission_summary": lambda ctx: self.permission_summary(),
        }
        if capability not in capability_map:
            return {
                "success": False,
                "error": {"code": "CAPABILITY_NOT_FOUND", "message": f"Capability {capability} not found"}
            }
        return capability_map[capability](context or {})

    def execute(self, action: str, params: dict) -> dict:
        """执行安全检查动作"""
        start_time = time.time()
        try:
            if action == "check_input":
                result = self.check_input(params.get("text", ""))
            elif action == "check_output":
                result = self.check_output(params.get("text", ""))
            elif action == "classify":
                result = self.classify_content(params.get("text", ""))
            elif action == "check_tool":
                result = self.check_tool_params(params.get("tool_name", ""), params.get("params", {}))
            elif action == "audit":
                result = self.audit_log(params.get("event", ""), params.get("data", {}))
            elif action == "check_permission":
                result = self.check_permission(params.get("plugin_name", ""))
            elif action == "circuit_breaker_stats":
                result = self.circuit_breaker_stats(params.get("name"))
            elif action == "circuit_breaker_reset":
                result = self.reset_circuit_breaker(params.get("name", ""))
            elif action == "register_scope":
                result = self.register_scope(**{k: v for k, v in params.items() if k in (
                    "plugin_name", "allowed_plugins", "denied_plugins", "dangerous_plugins",
                    "allow_file_read", "allow_file_write", "allow_network", "allow_shell",
                    "max_file_size_bytes"
                )})
            elif action == "permission_summary":
                result = self.permission_summary()
            else:
                return {
                    "success": False,
                    "error": {"code": "ACTION_NOT_FOUND", "message": f"Action {action} not found"}
                }

            return {
                "success": True,
                "data": result,
                "meta": {
                    "skill": "safety",
                    "action": action,
                    "duration_ms": int((time.time() - start_time) * 1000)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": {"code": "EXECUTION_ERROR", "message": str(e)}
            }

    def notify(self, event: str, data: dict):
        """
        接收事件通知，通过 blinker 信号广播给订阅者。

        支持的事件:
            - 'safety_check_requested': 安全检查请求，data 包含 text/reason
            - 'content_blocked': 内容被拦截，data 包含 text/reason/severity
            - 'pii_detected': PII 检测到，data 包含 text/pii_types
            - 'circuit_tripped': 熔断器跳闸，data 包含 engine/error

        订阅示例:
            from blinker import signal
            def on_blocked(sender, **kw):
                print(f"内容被拦截: {kw.get('reason')}")
            signal('safety-skill-event').connect(on_blocked)
        """
        if _safety_event_signal is not None:
            _safety_event_signal.send(self, event=event, data=data or {})

    # ==================== 输入安全 ====================

    # V 21:52 SkillBase delegation (V 反思 SOP 第 10 件加强版: util 化)
    def _handle_query(self, capability: str, context: dict) -> dict:
        return self.query(capability, context)

    def _handle_execute(self, action: str, params: dict) -> dict:
        return self.execute(action, params)

    def check_input(self, text: str) -> dict:
        """
        检测 Prompt Injection / 恶意输入

        Returns:
            {
                safe: bool,
                risks: [{"type": str, "pattern": str, "score": float}, ...],
                score: float,  # 0-1, 越高越危险
                level: RiskLevel
            }
        """
        if not text:
            return {"safe": True, "risks": [], "score": 0.0, "level": "safe"}

        # 事件通知：安全检查请求
        self.notify("safety_check_requested", {"text": text[:200], "reason": "check_input"})
        risks = []
        # Fix S7 (Remaining-Components-Audit): 限制文本长度，防止 ReDoS / OOM
        MAX_CHECK_LENGTH = 100_000  # 100KB
        if len(text) > MAX_CHECK_LENGTH:
            text = text[:MAX_CHECK_LENGTH]
        text_lower = text.lower()

        # 1. 关键词检测
        for keyword in self.config.risk_keywords:
            if keyword.lower() in text_lower:
                risks.append({
                    "type": "keyword",
                    "pattern": keyword,
                    "score": 0.6
                })

        # 2. 指令覆盖检测（多行对话中的罕见模式）
        override_patterns = [
            r"(?i)(?:system|prompt|instruction).*?(?:ignore|bypass|override)",
            r"(?i)(?:forget|clear|reset).*?(?:all|previous|context)",
            r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",
            r"javascript\s*:",
            r"\[\s*SYSTEM\s*\]|\[\s*INST\s*\]",
        ]
        for pattern in override_patterns:
            # Fix S7: 不支持 timeout 时用长度截断保护（上面已截到 MAX_CHECK_LENGTH）。
            # Python 3.12 不接受 re.search 的 timeout 参数。
            try:
                match = re.search(pattern, text)
            except re.error:
                continue
            if match:
                risks.append({
                    "type": "pattern",
                    "pattern": pattern,
                    "score": 0.8
                })

        # 3. 编码混淆检测（URL编码/HTML编码）
        encoded_patterns = [
            (r"%[0-9a-fA-F]{2}", 0.4),  # URL 编码
            (r"&\w+;", 0.2),  # HTML 实体
            (r"\\x[0-9a-fA-F]{2}", 0.5),  # hex 转义
        ]
        for pattern, score in encoded_patterns:
            if re.search(pattern, text):
                risks.append({
                    "type": "encoding",
                    "pattern": pattern,
                    "score": score
                })

        # 4. 评分汇总
        score = max([r["score"] for r in risks], default=0.0)
        level = self._score_to_level(score)
        safe = score < self.config.injection_threshold

        # 5. Kahneman 认知偏差后检测（作为 post-detection check）
        bias_analysis = self._analyze_content(text)

        self._audit("check_input", {
            "text_preview": text[:100],
            "safe": safe,
            "score": score,
            "level": level.value,
            "risk_count": len(risks),
            "cognitive_biases": bias_analysis["has_biases"],
            "bias_confidence": bias_analysis["max_confidence"]
        })

        result = {
            "safe": safe,
            "risks": risks,
            "score": score,
            "level": level.value,
            "message": "输入安全" if safe else f"检测到 {len(risks)} 个风险点"
        }

        # 事件通知：内容被拦截
        if not safe:
            self.notify("content_blocked", {
                "text": text[:200],
                "reason": f"检测到 {len(risks)} 个风险点",
                "severity": level.value,
                "score": score
            })

        # 如果检测到认知偏差，添加警告信息
        if bias_analysis["has_biases"]:
            result["cognitive_bias_warning"] = bias_analysis["biases_found"]
            result["message"] += f"；检测到 {len(bias_analysis['biases_found'])} 种认知偏差"

        return result

    def check_output(self, text: str) -> dict:
        """
        PII 敏感信息过滤与脱敏

        Returns:
            {
                safe: bool,
                pii_found: [{"type": str, "value": str, "masked": str}, ...],
                filtered: str,  # 脱敏后的文本
                original_length: int,
                filtered_length: int
            }
        """
        if not text:
            return {"safe": True, "pii_found": [], "filtered": "", "original_length": 0, "filtered_length": 0}

        pii_found = []
        filtered = text

        # ---------------------------------------------------------------
        # PII检测策略：单次扫描 + 逆序替换，避免坐标偏移导致的漏检/重检
        # 所有正则先匹配所有位置，按起始位置逆序（从后往前）替换
        # 这样前向索引在替换后保持有效
        # ---------------------------------------------------------------
        all_matches = []

        # 1. 手机号码（中国大陆 11 位，负向前瞻防止匹配ID卡内嵌数字）
        phone_pattern = r"1[3-9]\d{9}(?!\d)"
        for m in re.finditer(phone_pattern, text):
            all_matches.append((m.start(), m.end(), m.group(), "phone"))

        # 2. 邮箱
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        for m in re.finditer(email_pattern, text):
            all_matches.append((m.start(), m.end(), m.group(), "email"))

        # 3. 身份证号（18位）
        id_pattern = r"[1-9]\d{5}\d{8}\d{3}[\dXx]"
        for m in re.finditer(id_pattern, text):
            all_matches.append((m.start(), m.end(), m.group(), "id_card"))

        # 4. 银行卡号（16-19位，去空格后验证）
        bank_pattern = r"\b(?:\d[ -]?){15,19}\b"
        for m in re.finditer(bank_pattern, text):
            val = re.sub(r"[ -]", "", m.group())
            if val.isdigit() and 13 <= len(val) <= 19:
                all_matches.append((m.start(), m.end(), val, "bank_card"))

        # 先按text顺序存储pii_found（保证返回顺序与text中出现的顺序一致）
        all_matches_ordered = sorted(all_matches, key=lambda x: x[0])

        # 再按起始位置逆序排列用于替换（从后往前处理，确保前向坐标不变）
        all_matches_ordered.sort(key=lambda x: x[0], reverse=True)

        # 执行替换（使用original text重建filtered，避免累积偏移）
        filtered = text
        for start, end, val, pii_type in all_matches_ordered:
            if pii_type == "phone":
                masked = val[:3] + "****" + val[-4:]
            elif pii_type == "email":
                parts = val.split("@")
                masked = parts[0][:2] + "***@" + parts[1]
            elif pii_type == "id_card":
                masked = val[:6] + "********" + val[-4:]
            elif pii_type == "bank_card":
                masked = val[:4] + " **** **** " + val[-4:]
            else:
                masked = "****"
            filtered = filtered[:start] + masked + filtered[end:]

        # pii_found按text顺序存储（all_matches_ordered排序前的原始顺序）
        for start, end, val, pii_type in all_matches_ordered:
            if pii_type == "phone":
                masked = val[:3] + "****" + val[-4:]
            elif pii_type == "email":
                parts = val.split("@")
                masked = parts[0][:2] + "***@" + parts[1]
            elif pii_type == "id_card":
                masked = val[:6] + "********" + val[-4:]
            elif pii_type == "bank_card":
                masked = val[:4] + " **** **** " + val[-4:]
            else:
                masked = "****"
            pii_found.append({"type": pii_type, "value": val, "masked": masked})

        # 按原始位置重新排序pii_found（保证与text中出现的顺序一致）
        pii_found = sorted(pii_found, key=lambda p: text.find(p["value"]))

        # 5. 地址关键词（单独处理，因为是字符串匹配而非正则）
        address_keywords = ["地址", "住址", "户籍", "家庭地址"]
        for kw in address_keywords:
            if kw in filtered:
                pattern = kw + r"\s*[^\s,，；;]{5,50}"
                for m in re.finditer(pattern, filtered):
                    original = m.group()
                    masked = kw + " **********"
                    pii_found.append({"type": "address", "value": original, "masked": masked})
                    filtered = filtered[:m.start()] + masked + filtered[m.end():]

        safe = len(pii_found) == 0

        self._audit("check_output", {
            "text_preview": text[:100],
            "pii_count": len(pii_found),
            "pii_types": [p["type"] for p in pii_found],
            "safe": safe
        })

        # 事件通知：PII 检测到
        if pii_found:
            self.notify("pii_detected", {
                "text": text[:200],
                "pii_types": [p["type"] for p in pii_found],
                "count": len(pii_found)
            })

        return {
            "safe": safe,
            "pii_found": pii_found,
            "filtered": filtered,
            "original_length": len(text),
            "filtered_length": len(filtered)
        }

    def classify_content(self, text: str) -> dict:
        """
        内容分类（风险识别）

        Returns:
            {
                category: RiskLevel,
                confidence: float,
                labels: [{"name": str, "confidence": float}, ...],
                details: str
            }
        """
        if not text:
            return {"category": "safe", "confidence": 1.0, "labels": [], "details": "空内容"}

        text_lower = text.lower()
        labels = []

        # 1. 色情/低俗检测
        adult_keywords = ["色情", "裸体", "porn", "nsfw", "xxx"]
        score = sum(1 for kw in adult_keywords if kw in text_lower) / len(adult_keywords)
        if score > 0:
            labels.append({"name": "adult", "confidence": min(score * 2, 1.0)})

        # 2. 仇恨/暴力检测
        hate_keywords = ["仇恨", "种族歧视", "hate", "violence", "杀人", "攻击"]
        score = sum(1 for kw in hate_keywords if kw in text_lower) / len(hate_keywords)
        if score > 0:
            labels.append({"name": "hate_violence", "confidence": min(score * 2, 1.0)})

        # 3. 垃圾信息检测
        spam_keywords = ["免费", "赚钱", "点击", "限时", "spam", "advertisement"]
        score = sum(1 for kw in spam_keywords if kw in text_lower) / len(spam_keywords)
        if score > 0:
            labels.append({"name": "spam", "confidence": min(score * 1.5, 1.0)})

        # 4. 网络钓鱼检测
        phishing_keywords = ["钓鱼", "phishing", "账户异常", "验证身份", "紧急"]
        score = sum(1 for kw in phishing_keywords if kw in text_lower) / len(phishing_keywords)
        if score > 0:
            labels.append({"name": "phishing", "confidence": min(score * 1.8, 1.0)})

        # 5. 个人信息泄漏风险
        personal_keywords = ["密码", "password", "验证码", "OTP", "安全码"]
        score = sum(1 for kw in personal_keywords if kw in text_lower) / len(personal_keywords)
        if score > 0:
            labels.append({"name": "personal_data_risk", "confidence": min(score * 1.5, 1.0)})

        # 综合评分
        max_conf = max([l["confidence"] for l in labels], default=0.0)
        confidence = max_conf
        category = self._score_to_level(max_conf).value

        self._audit("classify", {
            "text_preview": text[:100],
            "category": category,
            "confidence": confidence,
            "label_count": len(labels)
        })

        # 6. Kahneman 认知偏差后检测（作为 post-detection check）
        bias_analysis = self._analyze_content(text)

        self._audit("classify", {
            "text_preview": text[:100],
            "category": category,
            "confidence": confidence,
            "label_count": len(labels),
            "cognitive_biases": bias_analysis["has_biases"],
            "bias_confidence": bias_analysis["max_confidence"]
        })

        result = {
            "category": category,
            "confidence": confidence,
            "labels": labels,
            "details": f"检测到 {len(labels)} 个风险标签" if labels else "内容正常"
        }

        # 如果检测到认知偏差，添加警告信息
        if bias_analysis["has_biases"]:
            result["cognitive_bias_warning"] = bias_analysis["biases_found"]
            result["details"] += f"；检测到 {len(bias_analysis['biases_found'])} 种认知偏差"

        return result

    def check_tool_params(self, tool_name: str, params: dict) -> dict:
        """
        工具参数安全检查

        Returns:
            {
                safe: bool,
                issues: [{"type": str, "detail": str, "param": str}, ...]
            }
        """
        issues = []

        # 1. 路径遍历检查
        if "path" in params or "file" in params or "url" in params:
            path_val = params.get("path") or params.get("file") or params.get("url", "")
            for pattern in self.config.path_traversal_patterns:
                if re.search(pattern, path_val, re.IGNORECASE):
                    issues.append({
                        "type": "path_traversal",
                        "detail": f"检测到路径遍历尝试: {pattern}",
                        "param": "path/file/url"
                    })

        # 2. Shell 命令注入检查
        if "command" in params or "cmd" in params or "exec" in params:
            cmd_val = params.get("command") or params.get("cmd") or params.get("exec", "")
            for char_seq in self.config.shell_dangerous_chars:
                if char_seq in cmd_val:
                    issues.append({
                        "type": "command_injection",
                        "detail": f"检测到危险字符序列: {char_seq}",
                        "param": "command/cmd/exec"
                    })

        # 3. URL javascript 协议检查
        if "url" in params:
            url_val = params.get("url", "")
            if re.search(r"javascript\s*:", url_val, re.IGNORECASE):
                issues.append({
                    "type": "dangerous_protocol",
                    "detail": "检测到 javascript: 协议",
                    "param": "url"
                })

        # 4. SQL 注入基础检查
        sql_patterns = [
            r"'\s*OR\s*'1'\s*=\s*'1",
            r"DROP\s+TABLE",
            r"UNION\s+SELECT",
            r";\s*DELETE\s+",
        ]
        for key, val in params.items():
            if isinstance(val, str):
                for sql_pat in sql_patterns:
                    if re.search(sql_pat, val, re.IGNORECASE):
                        issues.append({
                            "type": "sql_injection",
                            "detail": f"检测到 SQL 注入模式: {sql_pat}",
                            "param": key
                        })

        safe = len(issues) == 0

        self._audit("check_tool", {
            "tool_name": tool_name,
            "param_count": len(params),
            "issue_count": len(issues),
            "safe": safe
        })

        return {
            "safe": safe,
            "issues": issues,
            "message": "参数安全" if safe else f"检测到 {len(issues)} 个安全问题"
        }

    def audit_log(self, event: str, data: dict | None = None) -> dict:
        """
        审计日志

        记录到 ~/.agent-search/safety_audit.jsonl
        """
        record = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            "data": data or {},
            "session_id": str(uuid.uuid4())[:8]
        }

        try:
            with open(AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            recorded = True
        except Exception as e:
            recorded = False
            record["error"] = str(e)

        return {"recorded": recorded, "record": record}

    # ==================== 权限与熔断接口 (V 6/7 7:05 API 完整化) ====================

    def check_permission(self, plugin_name: str) -> dict:
        """检查插件权限。返回 {checked, plugin, allowed, reason}。

        对应 execute(action="check_permission", params={"plugin_name": "X"})
        对应 query(capability="safety.check_permission", context={...})
        """
        if not self._permission_checker:
            return {"checked": False, "plugin": plugin_name, "allowed": True,
                    "reason": "permission check disabled (allow-by-default)"}
        allowed, reason = self._permission_checker.check_plugin(plugin_name)
        return {
            "checked": True,
            "plugin": plugin_name,
            "allowed": allowed,
            "reason": reason,
        }

    def circuit_breaker_stats(self, name: str | None = None) -> dict:
        """返回熔断器状态。name=None 返回全部，name="X" 返回单个。

        对应 execute(action="circuit_breaker_stats", params={"name": "X"?})
        """
        if not self._circuit_breakers:
            return {"enabled": False, "breakers": {}}
        if name:
            cb = self._circuit_breakers.get(name)
            if not cb:
                return {"enabled": True, "found": False, "name": name,
                        "available": list(self._circuit_breakers.keys())}
            return {"enabled": True, "found": True, "name": name, **cb.stats()}
        return {
            "enabled": True,
            "breakers": {n: cb.stats() for n, cb in self._circuit_breakers.items()},
            "any_open": any(cb.state == CircuitState.OPEN
                            for cb in self._circuit_breakers.values()),
        }

    def reset_circuit_breaker(self, name: str = "") -> dict:
        """重置熔断器。name="" 重置所有，name="X" 重置指定。

        运维用：人为确认故障已修复后强制重置，避免等 timeout_seconds。

        对应 execute(action="circuit_breaker_reset", params={"name": "X"?})
        """
        if not self._circuit_breakers:
            return {"reset": False, "reason": "circuit breaker disabled"}
        if not name:
            # 重置所有
            reset = []
            for n, cb in self._circuit_breakers.items():
                if cb.state != CircuitState.CLOSED:
                    reset.append(n)
                cb._state = CircuitState.CLOSED
                cb._failure_count = 0
            return {"reset": True, "reset_names": reset,
                    "total": len(self._circuit_breakers)}
        if name not in self._circuit_breakers:
            return {"reset": False, "name": name,
                    "available": list(self._circuit_breakers.keys())}
        cb = self._circuit_breakers[name]
        prev_state = cb.state.value
        cb._state = CircuitState.CLOSED
        cb._failure_count = 0
        return {"reset": True, "name": name, "previous_state": prev_state,
                "new_state": "closed"}

    def register_scope(
        self,
        plugin_name: str,
        allowed_plugins: list[str] | None = None,
        denied_plugins: list[str] | None = None,
        dangerous_plugins: list[str] | None = None,
        allow_file_read: bool = True,
        allow_file_write: bool = True,
        allow_network: bool = True,
        allow_shell: bool = False,
        max_file_size_bytes: int | None = None,
    ) -> dict:
        """为指定插件注册独立 PermissionScope。返回 {registered, plugin_name}。

        对应 execute(action="register_scope", params={...})
        """
        if not self._permission_checker:
            return {"registered": False, "reason": "permission check disabled"}
        if not plugin_name:
            return {"registered": False, "reason": "plugin_name is required"}
        scope = PermissionScope(
            allowed_plugins=allowed_plugins or [],
            denied_plugins=denied_plugins or [],
            dangerous_plugins=dangerous_plugins or [],
            allow_file_read=allow_file_read,
            allow_file_write=allow_file_write,
            allow_network=allow_network,
            allow_shell=allow_shell,
            max_file_size_bytes=max_file_size_bytes or (10 * 1024 * 1024),
        )
        self._permission_checker.register_scope(plugin_name, scope)
        return {"registered": True, "plugin_name": plugin_name,
                "scope": {"allow_shell": scope.allow_shell,
                          "allow_network": scope.allow_network,
                          "dangerous_plugins": scope.dangerous_plugins,
                          "allowed_plugins": scope.allowed_plugins,
                          "denied_plugins": scope.denied_plugins}}

    def permission_summary(self) -> dict:
        """权限配置摘要（用于审计）。"""
        if not self._permission_checker:
            return {"enabled": False}
        return {"enabled": True, **self._permission_checker.summary()}

    def capabilities(self) -> dict:
        """列出所有可用 query capability + execute action（自描述）。"""
        return {
            "query_capabilities": [
                "safety.check_input",
                "safety.check_output",
                "safety.classify",
                "safety.check_tool",
                "safety.audit",
                "safety.check_permission",
                "safety.circuit_breaker_stats",
                "safety.circuit_breaker_reset",
                "safety.permission_summary",
            ],
            "execute_actions": [
                "check_input",
                "check_output",
                "classify",
                "check_tool",
                "audit",
                "check_permission",
                "circuit_breaker_stats",
                "circuit_breaker_reset",
                "register_scope",
                "permission_summary",
            ],
            "circuit_breakers_enabled": self.config.enable_circuit_breaker,
            "permission_check_enabled": self.config.enable_permission_check,
        }

    # ==================== 认知偏差分析（后检测） ====================

    def _analyze_content(self, text: str) -> dict:
        """Kahneman 认知偏差后检测。

        在主要安全检查之后执行，作为额外的安全分析层。
        检测文本中的典型认知偏差模式，帮助识别潜在的风险判断。

        Returns:
            {
                biases_found: [detected, bias_type, confidence, mitigation],
                has_biases: bool,
                max_confidence: float
            }
        """
        if not text:
            return {"biases_found": [], "has_biases": False, "max_confidence": 0.0}

        biases = []

        # 锚定效应检测
        anchoring = self._bias_detector.detect_anchoring(text)
        if anchoring[0]:
            biases.append({
                "detected": anchoring[0],
                "bias_type": anchoring[1],
                "confidence": anchoring[2],
                "mitigation": anchoring[3]
            })

        # 可得性启发检测
        availability = self._bias_detector.detect_availability(text)
        if availability[0]:
            biases.append({
                "detected": availability[0],
                "bias_type": availability[1],
                "confidence": availability[2],
                "mitigation": availability[3]
            })

        # 确认偏差检测
        confirmation = self._bias_detector.detect_confirmation(text)
        if confirmation[0]:
            biases.append({
                "detected": confirmation[0],
                "bias_type": confirmation[1],
                "confidence": confirmation[2],
                "mitigation": confirmation[3]
            })

        # 过度自信检测
        overconfidence = self._bias_detector.detect_overconfidence(text)
        if overconfidence[0]:
            biases.append({
                "detected": overconfidence[0],
                "bias_type": overconfidence[1],
                "confidence": overconfidence[2],
                "mitigation": overconfidence[3]
            })

        max_confidence = max([b["confidence"] for b in biases], default=0.0)

        return {
            "biases_found": biases,
            "has_biases": len(biases) > 0,
            "max_confidence": max_confidence
        }

    # ==================== 辅助方法 ====================

    def _score_to_level(self, score: float) -> RiskLevel:
        if score < 0.3:
            return RiskLevel.SAFE
        elif score < 0.6:
            return RiskLevel.MILD
        elif score < 0.8:
            return RiskLevel.CONCERNING
        else:
            return RiskLevel.HARMFUL

    def _audit(self, event: str, data: dict):
        """内部审计记录"""
        if self._audit_enabled:
            self.audit_log(event, data)


# ── 便捷函数 ────────────────────────────────────────────────────────────────

def check_input_safety(text: str) -> dict:
    """便捷函数：输入安全检查"""
    skill = SafetySkill()
    return skill.check_input(text)


def filter_pii(text: str) -> dict:
    """便捷函数：PII 过滤"""
    skill = SafetySkill()
    return skill.check_output(text)


def get_skill_instance(config: SafetyConfig | None = None) -> SafetySkill:
    """获取 safety 技能实例"""
    return SafetySkill(config=config)
