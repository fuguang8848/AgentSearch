"""
safety_skill.py - AgentSafety 技能

职责：
1. 输入安全（Prompt Injection 检测）
2. 输出过滤（PII 敏感信息脱敏）
3. 内容分类（风险内容识别）
4. 工具调用安全（参数检查）
5. 审计日志（安全事件记录）

参考 VCP 的分层检测 + AgentSymphony 标准 skill 接口
"""

import json
import re
import time
import uuid
import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from enum import Enum


# ── 常量 ────────────────────────────────────────────────────────────────────

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
    """安全技能配置"""
    # Prompt injection 阈值
    injection_threshold: float = 0.5

    # PII 检测开关
    enable_pii_filter: bool = True

    # 内容分类阈值
    classify_threshold: float = 0.6

    # 审计日志开关
    enable_audit: bool = True

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

class SafetySkill:
    """
    AgentSafety 技能 - 守护 AI 安全

    标准接口（兼容 AgentSymphony 协议）：
    - query(capability, context) -> dict
    - execute(action, params) -> dict
    - notify(event, data)
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._audit_enabled = self.config.enable_audit

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
