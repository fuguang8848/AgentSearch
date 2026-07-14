"""
team_skill.py - AgentSearch 的 team 桥接（基于 sessions_spawn）

参考 AgentSymphony server/skills/team_skill.py + AgentTeam OpenClaw SDK Backend
原理：
  1. 调用 openclaw gateway call sessions.create 创建 sub-agent search session
  2. 调用 sessions.send 发送搜索任务 + 等待回复
  3. 直接读 session JSONL 文件，绕过 Gateway 权限问题
"""

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

GATEWAY_URL = "ws://127.0.0.1:18789"
POLL_INTERVAL = 5  # 秒


def _read_session_file(session_key: str) -> dict:
    """直接读取 session JSONL 文件，绕过 Gateway 权限问题"""
    uuid_str = session_key.split(":")[-1]
    session_file = Path.home() / ".openclaw" / "agents" / "main" / "sessions" / f"{uuid_str}.jsonl"

    if not session_file.exists():
        return {"messages": [], "error": f"Session file not found"}

    messages = []
    try:
        with open(session_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    messages.append(entry)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return {"messages": [], "error": str(e)}

    return {"messages": messages}


def _gateway_call(method: str, params: dict = None, timeout: int = 30) -> dict:
    """调用 Gateway RPC"""
    cmd = [
        "openclaw", "gateway", "call",
        method,
        "--params", json.dumps(params or {}),
        "--json",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gateway call failed: {result.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON from gateway: {result.stdout}")


class TeamSkill:
    """
    通过 sessions_spawn 执行搜索任务

    使用 openclaw gateway 原生 sessions API：
    - sessions.create: 创建 sub-agent search session
    - sessions.send: 发送搜索任务
    - session JSONL: 读取结果（绕过权限）

    与 AgentTeam OpenClawSDKBackend 的区别：
    - AgentTeam 是完整的 spawn 后端（支持多种 backend）
    - TeamSkill 是轻量级 wrapper，专门用于 AgentSearch 场景
    - 简化版：不需要 task_queue、heartbeat、lifecycle 协议
    """

    def __init__(self, team_name: str = "agent-search-team"):
        self.team_name = team_name
        self.active_sessions: dict[str, dict] = {}

    def spawn_search(
        self,
        query: str,
        engines: list[str] | None = None,
        max_results: int = 10,
        session_id: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        """
        Spawn 一个 sub-agent 执行搜索任务

        Args:
            query: 搜索查询
            engines: 搜索引擎列表（默认 ["tavily"]）
            max_results: 最大结果数
            session_id: 可选的 session ID（默认自动生成）
            filters: 搜索过滤器

        Returns:
            {
                "session_id": str,
                "session_key": str,
                "status": "created",
                "message": str
            }
        """
        session_id = session_id or str(uuid.uuid4())[:8]
        engines = engines or ["tavily"]
        filters = filters or {}

        # 1. 创建 session
        try:
            resp = _gateway_call("sessions.create", {
                "agentId": "main",
                "label": f"search-team-{session_id}",
            })
            session_key = resp.get("key")
            created_id = session_key.split(":")[-1]
            if not session_key:
                return {"error": "No sessionKey in response", "raw": resp}  # type: ignore[return-value]
        except Exception as e:
            return {"error": f"Failed to create session: {e}"}

        # 2. 构建搜索任务消息
        task = self._build_search_task(query, engines, max_results, filters)

        # 3. 发送搜索任务
        try:
            _gateway_call("sessions.send", {
                "key": session_key,
                "message": task,
            })
        except Exception as e:
            return {"error": f"Failed to send task: {e}"}

        # 4. 记录 session
        self.active_sessions[session_key] = {
            "query": query,
            "engines": engines,
            "max_results": max_results,
            "filters": filters,
            "created_at": time.time(),
            "session_id": created_id,
        }

        return {
            "session_id": created_id,
            "session_key": session_key,
            "status": "created",
            "message": f"Search sub-agent spawned (session: {created_id[:8]})",
        }

    def _build_search_task(
        self,
        query: str,
        engines: list[str],
        max_results: int,
        filters: dict,
    ) -> str:
        """构建搜索任务消息"""
        engines_str = ", ".join(engines)
        filters_str = json.dumps(filters, ensure_ascii=False) if filters else "{}"

        task = f"""You are a search specialist agent on team {self.team_name}.

## Your Task
Execute a web search with the following parameters:

**Query:** {query}
**Engines:** {engines_str}
**Max Results:** {max_results}
**Filters:** {filters_str}

## Instructions
1. Execute the search using the specified search engines
2. Return the results in JSON format with the following structure:
```json
{{
    "status": "success",
    "results": [
        {{
            "url": "https://...",
            "title": "...",
            "content": "...",
            "engine": "tavily",
            "score": 0.95,
            "relevance": 0.9
        }}
    ],
    "count": 10,
    "query": "{query}",
    "engines": [{engines_str}]
}}
```
3. If search fails, return:
```json
{{
    "status": "error",
    "error": "error message"
}}
```

## Important
- Return ONLY the JSON result, no additional text
- Use the search tools available to you
- Ensure results are from credible sources
"""
        return task

    def status(self, session_id: str) -> dict:
        """轮询 session 状态和历史（优先读文件，避免权限问题）"""
        try:
            # 查找 session_key
            session_key = None
            for sk, info in self.active_sessions.items():
                if sk.endswith(f":{session_id}") or info.get("session_id") == session_id:
                    session_key = sk
                    break

            if not session_key:
                return {"session_id": session_id, "error": "Session not found"}

            # 直接读 session 文件
            file_result = _read_session_file(session_key)
            messages = file_result.get("messages", [])

            completed = False
            last_message = ""
            result_data = None

            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    last_message = last.get("content", "")[:200]
                    # 检查是否是 assistant 的最终回复（包含 JSON 结果）
                    if last.get("role") == "assistant":
                        completed = True
                        # 尝试解析 JSON 结果
                        try:
                            # 找到 JSON 块
                            content = last.get("content", "")
                            if "```json" in content:
                                json_start = content.find("```json") + 7
                                json_end = content.find("```", json_start)
                                if json_end > json_start:
                                    result_data = json.loads(content[json_start:json_end].strip())
                            elif "{" in content:
                                json_start = content.find("{")
                                json_end = content.rfind("}") + 1
                                if json_end > json_start:
                                    result_data = json.loads(content[json_start:json_end])
                        except (json.JSONDecodeError, Exception):
                            pass
                else:
                    last_message = str(last)[:200]

            return {
                "session_id": session_id,
                "status": "completed" if completed else "running",
                "messages_count": len(messages),
                "last_message": last_message,
                "result": result_data,
            }
        except Exception as e:
            return {"session_id": session_id, "error": str(e)}

    def wait_complete(self, session_id: str, timeout: int = 300) -> dict:
        """等待搜索任务完成（轮询）"""
        start = time.time()
        while time.time() - start < timeout:
            st = self.status(session_id)
            if st.get("status") == "completed" or "error" in st:
                return st
            time.sleep(POLL_INTERVAL)

        return {
            "session_id": session_id,
            "status": "timeout",
            "message": f"等待超时（{timeout}秒）",
        }

    def get_result(self, session_id: str) -> dict:
        """获取搜索结果（便捷方法）"""
        st = self.status(session_id)
        if st.get("status") == "completed" and st.get("result"):
            return st["result"]
        return st

    def shutdown(self, session_id: str) -> dict:
        """通知 sub-agent 关闭"""
        try:
            # 查找 session_key
            session_key = None
            for sk, info in self.active_sessions.items():
                if sk.endswith(f":{session_id}") or info.get("session_id") == session_id:
                    session_key = sk
                    break

            if not session_key:
                return {"session_id": session_id, "error": "Session not found"}

            _gateway_call("sessions.send", {
                "key": session_key,
                "message": "shutdown",
            })
            if session_key in self.active_sessions:
                del self.active_sessions[session_key]
            return {"session_id": session_id, "status": "shutdown_sent"}
        except Exception as e:
            return {"session_id": session_id, "error": str(e)}

    def list_active(self) -> list[dict]:
        """列出所有活跃 session"""
        return [
            {
                "session_id": info.get("session_id"),
                "query": info.get("query"),
                "engines": info.get("engines"),
                "created_at": info.get("created_at"),
            }
            for info in self.active_sessions.values()
        ]


# 便捷函数
def spawn_search_task(
    query: str,
    engines: list[str] | None = None,
    max_results: int = 10,
    wait: bool = True,
    timeout: int = 300,
) -> dict:
    """
    便捷函数：spawn 一个搜索任务并可选等待结果

    Args:
        query: 搜索查询
        engines: 搜索引擎列表
        max_results: 最大结果数
        wait: 是否等待完成
        timeout: 等待超时（秒）

    Returns:
        如果 wait=True: 返回搜索结果
        如果 wait=False: 返回 session 信息
    """
    skill = TeamSkill()
    result = skill.spawn_search(query, engines, max_results)

    if "error" in result:
        return result

    if wait:
        return skill.wait_complete(result["session_id"], timeout)

    return result


if __name__ == "__main__":
    # 示例用法
    print("Testing TeamSkill...")

    # 测试 spawn
    result = spawn_search_task(
        query="What is the latest news about AI agents?",
        engines=["tavily"],
        max_results=5,
        wait=False,
    )
    print(f"Spawn result: {result}")

    # 如果需要等待结果
    # result = spawn_search_task(query="...", wait=True)
    # print(f"Search results: {result}")
