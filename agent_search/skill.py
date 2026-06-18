"""
search 技能 - 搜索与信息获取

Agent Symphony 技能交响乐的信息获取中心
支持多引擎、爬虫、过滤、排序

增强版：集成真实搜索 API
- Tavily API (优先)
- Brave Search API
- 保留缓存机制与 symphony 协议兼容性
"""

import os
import time
import json
import hashlib
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# V 2026-06-04: agent_symphony 依赖改为可选，未装时用 mock fallback
# 这样 pip install -e . 可以独立装包（不用先装 AgentSymphony）
try:
    from agent_symphony.shared import (
        SharedContext,
        get_context,
    )
    _HAS_AGENT_SYMPHONY = True
except ImportError:
    # 没装 AgentSymphony 时，SharedContext 用本地 mock
    class SharedContext:  # type: ignore[no-redef]
        """Mock SharedContext（agent_symphony 未装时使用）"""
        def get_caller(self):
            class _Caller:
                caller_id = "v-mock"
            return _Caller()
        def set_search_query(self, q): pass
        def set_search_results(self, r): pass
    def get_context() -> SharedContext:  # type: ignore[no-redef]
        return SharedContext()
    _HAS_AGENT_SYMPHONY = False


class SearchEngine(Enum):
    """支持的搜索引擎"""
    TAVILY = "tavily"
    BRAVE = "brave"
    EXA = "exa"
    FIRECRAWL = "firecrawl"
    PERPLEXITY = "perplexity"
    MOCK = "mock"
    BING = "bing"  # V 端 search-v.py 集成（2026-06-04 V 加）
    GITHUB = "github"  # GitHub REST API 搜索（无需认证，2026-06-18 新增）


@dataclass
class SearchResult:
    """搜索结果"""
    url: str
    title: str
    content: str
    engine: str
    score: float = 0.0
    relevance: float = 0.0
    freshness: str = ""
    authority: float = 0.0
    cached: bool = False
    retrieved_at: float = field(default_factory=time.time)
    # 2026-06-18 新增：可信度评分（0-1，综合 freshness + authority + 引擎可靠性）
    trust_score: float = 0.0


@dataclass
class SearchConfig:
    """search 技能配置"""
    default_engines: list = field(default_factory=lambda: ["tavily"])
    max_results: int = 10
    relevance_threshold: float = 0.5
    cache_ttl: int = 3600  # 1小时
    timeout: int = 30  # 秒
    
    # API 配置
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    brave_api_key: str = field(default_factory=lambda: os.getenv("BRAVE_API_KEY", ""))
    exa_api_key: str = field(default_factory=lambda: os.getenv("EXA_API_KEY", ""))
    firecrawl_api_key: str = field(default_factory=lambda: os.getenv("FIRECRAWL_API_KEY", ""))
    perplexity_api_key: str = field(default_factory=lambda: os.getenv("PERPLEXITY_API_KEY", ""))
    # V 端 search-v.py 路径（Bing 引擎需要，可选）
    search_v_path: str = field(default_factory=lambda: os.getenv("SEARCH_V_PATH", ""))

    # 过滤器配置
    min_content_length: int = 100
    max_content_length: int = 10000
    languages: list = field(default_factory=lambda: ["zh", "en"])


class SearchAPIError(Exception):
    """搜索 API 错误"""
    def __init__(self, engine: str, message: str, status_code: int = 0):
        self.engine = engine
        self.message = message
        self.status_code = status_code
        super().__init__(f"[{engine}] {message}")


class SearchSkill:
    """
    Search 技能 - Agent Symphony 的信息获取中心
    
    核心职责：
    1. 多引擎搜索（支持 Tavily、Brave Search）
    2. 深度爬虫
    3. 智能过滤
    4. 结果排序
    5. 自动缓存
    """

    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()
        self._context: SharedContext = get_context()
        self._cache: dict[str, list[SearchResult]] = {}  # query_hash -> results
        self._cache_time: dict[str, float] = {}  # query_hash -> timestamp
        self._last_search_time: float = 0

    # ==================== 标准接口 ====================

    def query(self, capability: str, context: dict | None = None) -> dict:
        """
        查询技能能力
        """
        capability_map = {
            "search.execute": self._execute,
            "search.crawl": self._crawl,
            "search.filter": self._filter,
            "search.rank": self._rank,
            "search.cache": self._get_cache,
        }
        
        if capability not in capability_map:
            return {
                "success": False,
                "error": {
                    "code": "CAPABILITY_NOT_FOUND",
                    "message": f"Capability {capability} not found"
                }
            }
        
        return capability_map[capability](context or {})

    def execute(self, action: str, params: dict) -> dict:
        """
        执行动作
        """
        start_time = time.time()
        
        try:
            if action == "search":
                result = self._search(params)
            elif action == "crawl":
                result = self._crawl(params)
            elif action == "filter":
                result = self._filter(params)
            elif action == "rank":
                result = self._rank(params)
            elif action == "clear_cache":
                result = self._clear_cache(params)
            else:
                return {
                    "success": False,
                    "error": {
                        "code": "ACTION_NOT_FOUND",
                        "message": f"Action {action} not found"
                    }
                }
            
            return {
                "success": True,
                "data": result,
                "meta": {
                    "skill": "search",
                    "action": action,
                    "route_to": self._context.get_caller().caller_id if self._context.get_caller() else "user",
                    "duration_ms": int((time.time() - start_time) * 1000)
                }
            }
            
        except SearchAPIError as e:
            return {
                "success": False,
                "error": {
                    "code": "API_ERROR",
                    "engine": e.engine,
                    "message": e.message,
                    "status_code": e.status_code
                }
            }
        except Exception as e:
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
        pass  # 目前没有需要处理的事件

    # ==================== 核心方法 ====================

    def _search(self, params: dict) -> dict:
        """
        执行搜索
        """
        query = params.get("query", "")
        engines = params.get("engines", self.config.default_engines)
        max_results = params.get("max_results", self.config.max_results)
        filters = params.get("filters", {})
        
        if not query:
            return {
                "success": False,
                "error": {"code": "EMPTY_QUERY", "message": "Query is empty"}
            }
        
        # 检查缓存
        cache_key = self._get_cache_key(query, engines)
        cached_results = self._get_cached(cache_key)
        if cached_results:
            return {
                "results": [self._result_to_dict(r) for r in cached_results],
                "cached": True,
                "count": len(cached_results),
                "query": query
            }
        
        # 执行多引擎搜索
        all_results: list[SearchResult] = []
        used_engines = []
        
        for engine in engines:
            try:
                if engine == "tavily" and self.config.tavily_api_key:
                    results = self._search_tavily(query, max_results)
                    used_engines.append("tavily")
                elif engine == "brave" and self.config.brave_api_key:
                    results = self._search_brave(query, max_results)
                    used_engines.append("brave")
                elif engine == "bing" and self.config.search_v_path:
                    results = self._search_bing(query, max_results)
                    used_engines.append("bing")
                elif engine == "exa" and self.config.exa_api_key:
                    results = self._search_exa(query, max_results)
                    used_engines.append("exa")
                elif engine == "firecrawl" and self.config.firecrawl_api_key:
                    results = self._search_firecrawl(query, max_results)
                    used_engines.append("firecrawl")
                elif engine == "perplexity" and self.config.perplexity_api_key:
                    results = self._search_perplexity(query, max_results)
                    used_engines.append("perplexity")
                elif engine == "github":
                    # GitHub REST API 无需认证，直接可用
                    results = self._search_github(query, max_results)
                    used_engines.append("github")
                elif engine in ("tavily", "brave", "exa", "perplexity", "bing", "firecrawl"):
                    # API 未配置时跳过
                    continue
                else:
                    # 未知引擎或 mock
                    results = self._execute_search_mock(query, max_results)
                    used_engines.append("mock")
                
                all_results.extend(results)
            except SearchAPIError as e:
                # 单个引擎失败不影响其他引擎
                continue
        
        # 如果所有真实 API 都失败了，使用 mock
        if not all_results:
            all_results = self._execute_search_mock(query, max_results)
            used_engines = ["mock"]
        
        # 去重（按 URL）
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        # 过滤
        if filters:
            unique_results = self._apply_filters(unique_results, filters)

        # 排序
        unique_results = self._rank_results(unique_results, filters)

        # 限制数量
        unique_results = unique_results[:max_results]

        # 2026-06-18 新增：计算每条结果的可信度评分
        unique_results = self._compute_trust_scores(unique_results)

        # 缓存
        self._cache[cache_key] = unique_results
        self._cache_time[cache_key] = time.time()
        self._last_search_time = time.time()
        
        # 更新上下文
        self._context.set_search_query(query)
        self._context.set_search_results([
            {"url": r.url, "title": r.title, "content": r.content[:100]}
            for r in unique_results
        ])
        
        return {
            "results": [self._result_to_dict(r) for r in unique_results],
            "cached": False,
            "count": len(unique_results),
            "query": query,
            "engines": used_engines
        }

    # ==================== Tavily API ====================

    def _search_tavily(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        使用 Tavily API 执行搜索
        
        API 文档: https://docs.tavily.com/
        """
        import urllib.request
        import urllib.parse
        
        url = "https://api.tavily.com/search"
        
        payload = json.dumps({
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        })
        
        headers = {
            "Authorization": f"Bearer {self.config.tavily_api_key}",
            "Content-Type": "application/json"
        }
        
        req = urllib.request.Request(
            url,
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise SearchAPIError(
                "tavily",
                f"HTTP {e.code}: {error_body[:200]}",
                status_code=e.code
            )
        except urllib.error.URLError as e:
            raise SearchAPIError("tavily", f"URL Error: {str(e.reason)}")
        
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                engine="tavily",
                score=item.get("score", 0.0),
                relevance=item.get("score", 0.0),
                freshness=item.get("published_date", ""),
                authority=0.5  # Tavily 不提供权威性评分
            ))
        
        return results

    # ==================== Brave Search API ====================

    def _search_brave(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        使用 Brave Search API 执行搜索
        
        API 文档: https://api.search.brave.com/app/documentation/web-search/get-started
        """
        import urllib.request
        import urllib.parse
        
        base_url = "https://api.search.brave.com/res/v1/web/search"
        
        params = urllib.parse.urlencode({
            "q": query,
            "count": min(max_results, 20),  # Brave 最多 20
        })
        
        url = f"{base_url}?{params}"
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.config.brave_api_key
        }
        
        req = urllib.request.Request(url, headers=headers, method="GET")
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise SearchAPIError(
                "brave",
                f"HTTP {e.code}: {error_body[:200]}",
                status_code=e.code
            )
        except urllib.error.URLError as e:
            raise SearchAPIError("brave", f"URL Error: {str(e.reason)}")
        
        results = []
        web_results = data.get("web", {}).get("results", [])
        
        for item in web_results:
            # Brave 提供 age 和 meta_url 信息
            age = item.get("age", "")
            
            results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                content=item.get("description", ""),
                engine="brave",
                score=0.0,  # Brave 不直接提供分数
                relevance=0.0,
                freshness=age,
                authority=0.5  # Brave 不提供权威性评分
            ))

        return results

    # ==================== Exa Search API ====================

    def _search_exa(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        使用 Exa Search API 执行搜索

        API 文档: https://docs.exa.ai/
        """
        import urllib.request
        import urllib.parse

        url = "https://api.exa.ai/search"

        payload = json.dumps({
            "query": query,
            "numResults": min(max_results, 100),
            "includeArticles": True,
        })

        headers = {
            "Authorization": f"Bearer {self.config.exa_api_key}",
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(
            url,
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise SearchAPIError(
                "exa",
                f"HTTP {e.code}: {error_body[:200]}",
                status_code=e.code
            )
        except urllib.error.URLError as e:
            raise SearchAPIError("exa", f"URL Error: {str(e.reason)}")

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                content=item.get("snippet", ""),
                engine="exa",
                score=item.get("score", 0.0),
                relevance=item.get("score", 0.0),
                freshness=item.get("publishedDate", ""),
                authority=0.5  # Exa 不直接提供权威性评分
            ))

        return results

    # ==================== Firecrawl API ====================

    def _search_firecrawl(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        使用 Firecrawl API 执行搜索（通过 /search 端点）

        API 文档: https://docs.firecrawl.dev/
        """
        import urllib.request
        import urllib.parse

        base_url = "https://api.firecrawl.dev/v0/search"
        params = urllib.parse.urlencode({
            "query": query,
            "limit": min(max_results, 20),
        })

        url = f"{base_url}?{params}"

        headers = {
            "Authorization": f"Bearer {self.config.firecrawl_api_key}",
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise SearchAPIError(
                "firecrawl",
                f"HTTP {e.code}: {error_body[:200]}",
                status_code=e.code
            )
        except urllib.error.URLError as e:
            raise SearchAPIError("firecrawl", f"URL Error: {str(e.reason)}")

        results = []
        # Firecrawl v0/search 返回 { "data": [ { "url", "title", "description", ... } ] }
        for item in data.get("data", []):
            results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                content=item.get("description", ""),
                engine="firecrawl",
                score=0.0,  # Firecrawl 不直接提供分数
                relevance=0.0,
                freshness=item.get("publishedDate", ""),
                authority=0.5  # Firecrawl 不提供权威性评分
            ))

        return results

    # ==================== Perplexity API ====================

    def _search_perplexity(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        使用 Perplexity API 执行搜索（Sonar 模型）

        API 文档: https://docs.perplexity.ai/
        """
        import urllib.request
        import urllib.parse

        url = "https://api.perplexity.ai/chat/completions"

        payload = json.dumps({
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate, real-time information."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": 1000,
            "return_citations": True,
        })

        headers = {
            "Authorization": f"Bearer {self.config.perplexity_api_key}",
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(
            url,
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise SearchAPIError(
                "perplexity",
                f"HTTP {e.code}: {error_body[:200]}",
                status_code=e.code
            )
        except urllib.error.URLError as e:
            raise SearchAPIError("perplexity", f"URL Error: {str(e.reason)}")

        results = []
        # Perplexity 返回 citations 列表，格式为 URL
        citations = data.get("citations", [])
        # 也从 content 中解析出引用的 URL
        for idx, citation_url in enumerate(citations[:max_results]):
            results.append(SearchResult(
                url=citation_url,
                title=f"Result {idx + 1}",
                content="",  # Perplexity chat 格式没有独立 snippet
                engine="perplexity",
                score=1.0 - (idx * 0.1),  # 按引用顺序递减
                relevance=1.0 - (idx * 0.1),
                freshness="",
                authority=0.5
            ))

        # 如果没有 citations，至少记录一次成功的搜索
        if not results:
            results.append(SearchResult(
                url="https://perplexity.ai/search",
                title=f"Perplexity Sonar Search: {query}",
                content=data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                engine="perplexity",
                score=0.5,
                relevance=0.5,
                freshness="",
                authority=0.5
            ))

        return results[:max_results]

    # ==================== GitHub REST API 搜索（2026-06-18 新增） ====================

    def _search_github(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        使用 GitHub REST API 搜索公开仓库（无需认证）。

        API 文档: https://docs.github.com/en/rest/search

        特点：
        - 无需 GitHub Token，直接可用（公开仓库）
        - 按 stars 排序，返回最相关的项目
        - 每个结果包含 stars/forks/language/license/topics
        - 验证：所有引用的 stars 必须实时 curl 验证
        """
        import urllib.request
        import urllib.parse
        import json

        # 搜索仓库（按 stars 降序）
        search_url = "https://api.github.com/search/repositories"
        params = urllib.parse.urlencode({
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_results, 30),  # GitHub 限制最大30
        })

        url = f"{search_url}?{params}"

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Hermes-Agent-Search/1.0",  # GitHub 要求 User-Agent
        }

        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")[:200] if e.fp else ""
            raise SearchAPIError(
                "github",
                f"HTTP {e.code}: {error_body}",
                status_code=e.code
            )
        except urllib.error.URLError as e:
            raise SearchAPIError("github", f"URL Error: {str(e.reason)}")

        results = []
        for item in data.get("items", [])[:max_results]:
            # 计算权威性分数：基于 stars + forks（归一化到 0-1）
            stars = item.get("stargazers_count", 0)
            forks = item.get("forks_count", 0)
            # stars 权重更高：stars*1.0 + forks*0.3，归一化（假设 1000 stars + 500 forks 为满分）
            authority = min((stars + forks * 0.3) / 1300.0, 1.0)

            results.append(SearchResult(
                url=item.get("html_url", ""),
                title=f"{item.get('full_name', '')}",
                content=item.get("description", "") or "",
                engine="github",
                score=float(item.get("stargazers_count", 0)),
                relevance=0.9,  # GitHub 搜索本身已按相关性排序
                freshness=item.get("updated_at", "")[:10],  # 取日期部分 YYYY-MM-DD
                authority=round(authority, 3),  # 0-1 的权威性分数
            ))

        return results

    # ==================== Bing (V 端 search-v.py 集成) ====================

    def _search_bing(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        使用 V 端 search-v.py (Bing HTML) 搜索

        集成方式: importlib 动态加载（不污染 sys.path）
        依赖: httpx + lxml（在 search-v.py 里；Bing 引擎需 pip install agent-search[bing]）

        配置: SearchConfig.search_v_path = "/path/to/search-v.py"
        环境变量: SEARCH_V_PATH

        API: search-v.py 暴露 search_bing(query, limit) -> (results, error)

        V 2026-06-04 重构: 加 mod cache（避免每次 reload） + 精准异常
        """
        # 顶层导入（不在函数体里）
        import importlib.util
        import os

        search_v_path = self.config.search_v_path
        if not search_v_path or not os.path.exists(search_v_path):
            raise SearchAPIError(
                engine="bing",
                message=f"search_v_path not set or file missing: {search_v_path}",
            )

        # Cache 加载的 module（key = 绝对路径，避免同一 skill 多次 reload）
        cache_key = os.path.abspath(search_v_path)
        if not hasattr(self, "_v_search_v_cache"):
            self._v_search_v_cache = {}
        mod = self._v_search_v_cache.get(cache_key)
        if mod is None:
            try:
                spec = importlib.util.spec_from_file_location("v_search_v", search_v_path)
                if spec is None or spec.loader is None:
                    raise SearchAPIError(
                        engine="bing",
                        message=f"importlib spec failed for {search_v_path}",
                    )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                self._v_search_v_cache[cache_key] = mod
            except SearchAPIError:
                raise
            except (ImportError, SyntaxError, AttributeError) as e:
                # 精准异常（不抓 KeyboardInterrupt/SystemExit/其他）
                raise SearchAPIError(
                    engine="bing",
                    message=f"importlib load failed: {type(e).__name__}: {e}",
                )

        try:
            raw_results, err = mod.search_bing(query, limit=max_results)
        except AttributeError as e:
            raise SearchAPIError(
                engine="bing",
                message=f"search_bing() missing in {search_v_path}: {e}",
            )
        if err:
            raise SearchAPIError(engine="bing", message=err)

        results: list[SearchResult] = []
        for item in raw_results[:max_results]:
            results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                engine="bing",
                score=0.0,  # Bing HTML 不直接提供分数
                relevance=0.6,  # 中等默认（无评分）
                freshness="",  # search-v.py 不解析日期
                authority=0.5,  # 中等默认
            ))

        return results

    # ==================== Mock 实现（备选） ====================

    def _execute_search_mock(self, query: str, max_results: int) -> list[SearchResult]:
        """
        Mock 搜索结果（当没有配置 API 时使用）
        """
        mock_results = [
            SearchResult(
                url="https://example.com/article1",
                title=f"关于 {query} 的文章 1",
                content=f"这是关于 {query} 的详细内容，包含多个方面的信息...",
                engine="mock",
                score=0.95,
                relevance=0.9,
                freshness="2024-01",
                authority=0.8
            ),
            SearchResult(
                url="https://example.com/article2",
                title=f"{query} 详解",
                content=f"深入分析 {query} 的各个方面，包括原理、实践和案例...",
                engine="mock",
                score=0.88,
                relevance=0.85,
                freshness="2024-02",
                authority=0.75
            ),
            SearchResult(
                url="https://example.com/article3",
                title=f"如何正确理解 {query}",
                content=f"本指南帮助你理解 {query} 的核心概念和应用场景...",
                engine="mock",
                score=0.82,
                relevance=0.78,
                freshness="2024-03",
                authority=0.7
            ),
        ]
        
        return mock_results[:max_results]

    # 保留旧方法名以保持兼容性
    def _execute_search(self, query: str, engines: list, max_results: int) -> list[SearchResult]:
        """兼容旧接口"""
        return self._execute_search_mock(query, max_results)

    def _execute(self, context: dict) -> dict:
        """执行搜索（接口兼容）"""
        return self._search(context)

    # ==================== 爬虫接口 ====================

    def _crawl(self, params: dict) -> dict:
        """
        深度爬取
        
        支持：
        - 直接返回模拟内容（无 API 时）
        - Tavily Extract API（未来扩展）
        """
        url = params.get("url", "")
        
        if not url:
            return {
                "success": False,
                "error": {"code": "EMPTY_URL", "message": "URL is empty"}
            }
        
        # 优先尝试 Tavily Extract（如果配置了 API）
        if self.config.tavily_api_key:
            try:
                return self._crawl_tavily(url)
            except SearchAPIError:
                pass  # 降级到模拟
        
        # 模拟爬取
        return {
            "url": url,
            "content": f"从 {url} 爬取的内容...",
            "title": "爬取的页面标题",
            "links": ["https://example.com/link1", "https://example.com/link2"],
            "crawled_at": time.time()
        }

    def _crawl_tavily(self, url: str) -> dict:
        """
        使用 Tavily Extract API 深度爬取页面
        
        API: POST https://api.tavily.com/extract
        """
        import urllib.request
        import urllib.error
        
        api_url = "https://api.tavily.com/extract"
        
        payload = json.dumps({
            "urls": [url]
        })
        
        headers = {
            "Authorization": f"Bearer {self.config.tavily_api_key}",
            "Content-Type": "application/json"
        }
        
        req = urllib.request.Request(
            api_url,
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise SearchAPIError(
                "tavily",
                f"Extract HTTP {e.code}: {error_body[:200]}",
                status_code=e.code
            )
        except urllib.error.URLError as e:
            raise SearchAPIError("tavily", f"Extract URL Error: {str(e.reason)}")
        
        results = data.get("results", [])
        if results:
            result = results[0]
            return {
                "url": url,
                "content": result.get("raw_content", ""),
                "title": result.get("title", ""),
                "links": [],  # Tavily extract 不返回链接
                "crawled_at": time.time()
            }
        
        return {
            "url": url,
            "content": "",
            "title": "",
            "links": [],
            "crawled_at": time.time()
        }

    # ==================== 过滤与排序 ====================

    def _filter(self, params: dict) -> dict:
        """
        过滤搜索结果
        """
        results_data = params.get("results", [])
        filters = params.get("filters", {})
        
        # 转换为 SearchResult 对象
        results = [
            SearchResult(**r) if isinstance(r, dict) else r
            for r in results_data
        ]
        
        # 应用过滤
        filtered = self._apply_filters(results, filters)
        
        return {
            "results": [self._result_to_dict(r) for r in filtered],
            "count": len(filtered),
            "original_count": len(results)
        }

    def _apply_filters(self, results: list[SearchResult], filters: dict) -> list[SearchResult]:
        """
        应用过滤规则
        """
        filtered = results
        
        # 相关性过滤
        if "relevance" in filters:
            min_relevance = filters["relevance"]
            filtered = [r for r in filtered if r.relevance >= min_relevance]
        
        # 语言过滤
        if "languages" in filters:
            languages = filters["languages"]
            # 简化实现
            filtered = [r for r in filtered]  # TODO: 实际检查语言
        
        # 时效性过滤
        if "freshness" in filters:
            # 简化实现
            pass
        
        # 权威性过滤
        if "authority" in filters:
            min_authority = filters["authority"]
            filtered = [r for r in filtered if r.authority >= min_authority]
        
        return filtered

    def _rank(self, params: dict) -> dict:
        """
        排序搜索结果
        """
        results_data = params.get("results", [])
        criteria = params.get("criteria", {})
        
        results = [
            SearchResult(**r) if isinstance(r, dict) else r
            for r in results_data
        ]
        
        ranked = self._rank_results(results, criteria)
        
        return {
            "results": [self._result_to_dict(r) for r in ranked],
            "count": len(ranked)
        }

    def _rank_results(self, results: list[SearchResult], criteria: dict) -> list[SearchResult]:
        """
        对搜索结果排序
        """
        weights = {
            "relevance": criteria.get("relevance_weight", 0.4),
            "freshness": criteria.get("freshness_weight", 0.3),
            "authority": criteria.get("authority_weight", 0.2),
            "score": criteria.get("score_weight", 0.1),
        }

        def _freshness_to_number(freshness: str) -> float:
            """将 freshness 字符串转为 0-1 数值（与 _compute_trust_scores 保持一致）"""
            if not freshness:
                return 0.3
            try:
                if len(freshness) >= 10:
                    import datetime
                    dt = datetime.datetime.strptime(freshness[:10], "%Y-%m-%d")
                    days_ago = (datetime.datetime.now() - dt).days
                    if days_ago <= 30:
                        return 1.0
                    elif days_ago <= 90:
                        return 0.8
                    elif days_ago <= 180:
                        return 0.6
                    elif days_ago <= 365:
                        return 0.4
                    else:
                        return 0.2
                return 0.3
            except (ValueError, TypeError):
                return 0.3

        def calculate_rank_score(r: SearchResult) -> float:
            freshness_num = _freshness_to_number(r.freshness)
            return (
                r.relevance * weights["relevance"] +
                freshness_num * weights["freshness"] +
                r.authority * weights["authority"] +
                r.score * weights["score"]
            )

        return sorted(results, key=calculate_rank_score, reverse=True)

    # ==================== 缓存接口 ====================

    def _get_cache(self, context: dict) -> dict:
        """获取缓存"""
        query = context.get("query", "")
        engines = context.get("engines", self.config.default_engines)
        cache_key = self._get_cache_key(query, engines)
        
        cached = self._get_cached(cache_key)
        
        return {
            "cached": cached is not None,
            "results": [self._result_to_dict(r) for r in cached] if cached else [],
            "cache_key": cache_key
        }

    def _clear_cache(self, params: dict) -> dict:
        """清空缓存"""
        count = len(self._cache)
        self._cache = {}
        self._cache_time = {}
        return {"cleared": True, "count": count}

    # ==================== 辅助方法 ====================

    def _get_cache_key(self, query: str, engines: list) -> str:
        """生成缓存 key"""
        content = f"{query}:{','.join(sorted(engines))}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> list[SearchResult] | None:
        """获取缓存结果"""
        if cache_key not in self._cache:
            return None
        
        cached = self._cache[cache_key]
        cache_time = self._cache_time.get(cache_key, 0)
        
        # 检查是否过期
        if time.time() - cache_time > self.config.cache_ttl:
            del self._cache[cache_key]
            if cache_key in self._cache_time:
                del self._cache_time[cache_key]
            return None
        
        # 标记为缓存
        for r in cached:
            r.cached = True
        
        return cached

    # ==================== 2026-06-18 新增：可信度评分 ====================

    def _compute_trust_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        计算每条搜索结果的可信度评分（0-1）。

        计算公式：
        trust_score = w_authority * authority + w_freshness * freshness_score + w_engine * engine_score

        其中：
        - authority: 0-1（GitHub stars/forks 归一化）
        - freshness_score: 根据 freshness 字符串计算（越新越好）
        - engine_score: 引擎可靠性（github=0.9, perplexity=0.8, tavily=0.7, brave=0.7, exa=0.7, firecrawl=0.6, bing=0.5, mock=0.1）

        默认权重: authority=0.4, freshness=0.3, engine=0.3
        """
        import datetime

        # 引擎可靠性评分
        ENGINE_SCORES = {
            "github": 0.9,   # 有真实 stars/forks 数据，难造假
            "perplexity": 0.8,
            "tavily": 0.7,
            "brave": 0.7,
            "exa": 0.7,
            "firecrawl": 0.6,
            "bing": 0.5,
            "mock": 0.1,     # mock 数据完全不可信
        }

        W_AUTHORITY = 0.4
        W_FRESHNESS = 0.3
        W_ENGINE = 0.3

        def _parse_freshness(freshness: str) -> float:
            """将 freshness 字符串转为 0-1 分数"""
            if not freshness:
                return 0.3  # 无日期信息，中等可信
            try:
                # 尝试解析 YYYY-MM-DD 格式
                if len(freshness) >= 10:
                    date_str = freshness[:10]
                    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    days_ago = (datetime.datetime.now() - dt).days
                    # 0-30天=1.0, 30-90天=0.8, 90-180天=0.6, 180-365天=0.4, 1年+=0.2
                    if days_ago <= 30:
                        return 1.0
                    elif days_ago <= 90:
                        return 0.8
                    elif days_ago <= 180:
                        return 0.6
                    elif days_ago <= 365:
                        return 0.4
                    else:
                        return 0.2
                return 0.3
            except (ValueError, TypeError):
                return 0.3  # 解析失败，中等可信

        for result in results:
            engine_s = ENGINE_SCORES.get(result.engine, 0.5)
            freshness_s = _parse_freshness(result.freshness)
            # authority 已经是 0-1 的值（来自 GitHub 搜索）
            auth_s = result.authority if result.authority else 0.3

            trust = W_AUTHORITY * auth_s + W_FRESHNESS * freshness_s + W_ENGINE * engine_s
            result.trust_score = round(min(trust, 1.0), 3)

        # 按 trust_score 降序重排
        results.sort(key=lambda r: r.trust_score, reverse=True)
        return results

    def _result_to_dict(self, result: SearchResult) -> dict:
        """转换结果为字典"""
        return {
            "url": result.url,
            "title": result.title,
            "content": result.content,
            "engine": result.engine,
            "score": result.score,
            "relevance": result.relevance,
            "freshness": result.freshness,
            "authority": result.authority,
            "cached": result.cached,
            "retrieved_at": result.retrieved_at,
            "trust_score": result.trust_score,  # 2026-06-18 新增
        }


def get_skill_instance(config: SearchConfig | None = None) -> SearchSkill:
    """获取 search 技能实例"""
    return SearchSkill(config=config)
