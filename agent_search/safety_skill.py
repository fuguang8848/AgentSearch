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
import logging
from pathlib import Path
from typing import Any, Optional
from .skill_base import SkillBase
from dataclasses import dataclass, field
from enum import Enum


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
        if self._state == CircuitState.CLOSED:
            return
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                logger.warning(f"[CircuitBreaker] {self.name} → HALF_OPEN")
            else:
                raise RuntimeError(
                    f"Circuit '{self.name}' is OPEN. "
                    f"Retry in {self.timeout_seconds - (time.time() - self._last_failure_time):.1f}s"
                )

    def _record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(f"[CircuitBreaker] {self.name} HALF_OPEN→OPEN (failure)")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"[CircuitBreaker] {self.name} CLOSED→OPEN (threshold reached)")

    def _record_success(self):
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
            "safety.check_permission": lambda ctx: self._check_permission(ctx.get("plugin_name", "")),
            "safety.circuit_breaker_stats": lambda ctx: self._circuit_breaker_stats(),
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
        """接收事件通知（目前用于自动检查）"""
        pass

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

        risks = []
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
            if re.search(pattern, text):
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

        self._audit("check_input", {
            "text_preview": text[:100],
            "safe": safe,
            "score": score,
            "level": level.value,
            "risk_count": len(risks)
        })

        return {
            "safe": safe,
            "risks": risks,
            "score": score,
            "level": level.value,
            "message": "输入安全" if safe else f"检测到 {len(risks)} 个风险点"
        }

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

        # 1. 手机号码（中国大陆 11 位，必须有边界防止匹配 ID 卡内嵌数字）
        phone_pattern = r"1[3-9]\d{9}(?!\d)"
        # 先收集所有匹配位置（防止重复处理）
        phone_matches = [(m.group(), m.start(), m.end()) for m in re.finditer(phone_pattern, filtered)]
        for phone_val, start, end in reversed(phone_matches):  # 逆序，从后往前替换
            masked = phone_val[:3] + "****" + phone_val[-4:]
            pii_found.append({"type": "phone", "value": phone_val, "masked": masked})
            filtered = filtered[:start] + masked + filtered[end:]

        # 重新匹配（内容已变）
        for pii in pii_found:
            if pii["masked"] in filtered:
                continue  # 已处理

        # 2. 邮箱
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        for match in re.finditer(email_pattern, filtered):
            email = match.group()
            parts = email.split("@")
            masked = parts[0][:2] + "***@" + parts[1]
            pii_found.append({"type": "email", "value": email, "masked": masked})
            filtered = filtered[:match.start()] + masked + filtered[match.end():]

        # 3. 身份证号（18位，最后一位可以是数字或X）
        # 分两部分：\d{14} 覆盖前段，\d{3}[\dXx] 精确覆盖序列+校验位
        id_pattern = r"[1-9]\d{5}\d{8}\d{3}[\dXx]"
        for match in re.finditer(id_pattern, filtered):
            masked = match.group()[:6] + "********" + match.group()[-4:]
            pii_found.append({"type": "id_card", "value": match.group(), "masked": masked})
            filtered = filtered[:match.start()] + masked + filtered[match.end():]

        # 4. 银行卡号（16-19 位）
        bank_pattern = r"\b(?:\d[ -]?){15,19}\b"
        for match in re.finditer(bank_pattern, filtered):
            val = re.sub(r"[ -]", "", match.group())
            if val.isdigit() and 13 <= len(val) <= 19:
                masked = val[:4] + " **** **** " + val[-4:]
                pii_found.append({"type": "bank_card", "value": val, "masked": masked})
                filtered = filtered[:match.start()] + masked + filtered[match.end():]

        # 5. 地址（简单关键词检测）
        address_keywords = ["地址", "住址", "户籍", "家庭地址"]
        for kw in address_keywords:
            if kw in filtered:
                # 简单掩码：关键词后跟的连续非空白字符
                pattern = kw + r"\s*[^\s,，；;]{5,50}"
                for m in re.finditer(pattern, filtered):
                    original = m.group()
                    # 保留关键词，掩码内容
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

        return {
            "category": category,
            "confidence": confidence,
            "labels": labels,
            "details": f"检测到 {len(labels)} 个风险标签" if labels else "内容正常"
        }

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

    # ==================== 权限与熔断接口 ====================

    def _check_permission(self, plugin_name: str) -> dict:
        """检查插件权限。"""
        if not self._permission_checker:
            return {"checked": False, "reason": "permission check disabled"}
        allowed, reason = self._permission_checker.check_plugin(plugin_name)
        return {
            "checked": True,
            "plugin": plugin_name,
            "allowed": allowed,
            "reason": reason,
        }

    def _circuit_breaker_stats(self) -> dict:
        """返回所有熔断器状态。"""
        if not self._circuit_breakers:
            return {"enabled": False}
        return {
            "enabled": True,
            "breakers": {name: cb.stats() for name, cb in self._circuit_breakers.items()},
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
