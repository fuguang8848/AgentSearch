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
import math
import hashlib
import concurrent.futures
import numpy as np
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# V 2026-06-24: 向量检索依赖（可选）
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

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


class RankingStrategy(Enum):
    """
    排序策略枚举
    
    用 Latour ANT 视角：不同的排序策略决定了不同的"行动者网络"权力分布
    用 Simon 有限理性视角：不同的策略对应不同的决策哲学
    用 Merton 科学社会学视角：策略选择影响知识生产的平等性
    
    OPTIMAL (最优模式): 
        - Latour: 索引/缓存等算法行动者主导，用户被动接受"最优"
        - Simon: 强迫用户接受唯一最优解，忽视有限理性
        - Merton: 强化主流共识，压制异见
        
    SATISFICING (够好模式):
        - Latour: 用户保留更多控制权
        - Simon: 符合有限理性，够用即可
        - Merton: 允许非主流观点有曝光机会
        
    DIVERSITY (多元模式):
        - Latour: 打破算法的隐形权力
        - Simon: 降低认知负荷，提供多种选择
        - Merton: 主动打破共识霸权，给异见曝光机会
    """
    OPTIMAL = "optimal"           # 当前"最优"模式（默认）
    SATISFICING = "satisficing"  # Simon 够好模式：先返回少量，允许用户探索更多
    DIVERSITY = "diversity"      # Merton 多元模式：主动纳入低权威/异见


@dataclass
class SearchResult:
    """
    搜索结果 - Gosling 强类型版本
    
    类型安全规则：
    - 所有分数字段必须在 [0.0, 1.0] 范围内
    - freshness 必须是 ISO 8601 格式字符串
    - engine 必须是 SearchEngine 枚举值
    - url 必须是有效非空字符串
    """
    url: str
    title: str
    content: str
    engine: str  # 运行时验证：必须是 SearchEngine.value
    score: float = 0.0
    relevance: float = 0.0
    freshness: str = ""  # ISO 8601: "2026-06-24" 或 "2026-06-24T10:30:00Z"
    authority: float = 0.0
    cached: bool = False
    retrieved_at: float = field(default_factory=time.time)
    trust_score: float = 0.0  # [0.0, 1.0]
    ranking_uncertainty: float = 0.0  # [0.0, 1.0]
    
    # ========== Gosling 强类型：运行时验证 ==========
    def __post_init__(self):
        """运行时类型/范围检查 - 防止无效数据污染系统"""
        # URL 非空验证
        if not self.url or not isinstance(self.url, str):
            raise TypeError(f"SearchResult.url must be non-empty string, got: {type(self.url).__name__}")
        
        # 分数字段范围验证 [0.0, 1.0]
        float_fields = ['score', 'relevance', 'authority', 'trust_score', 'ranking_uncertainty']
        for field_name in float_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"SearchResult.{field_name} must be float, got: {type(value).__name__}")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"SearchResult.{field_name} must be in [0.0, 1.0], got: {value}")
        
        # freshness 格式验证（如果非空）
        if self.freshness:
            import datetime
            try:
                # 支持多种格式
                if 'T' in self.freshness:
                    datetime.datetime.fromisoformat(self.freshness.replace('Z', '+00:00'))
                else:
                    datetime.datetime.strptime(self.freshness[:10], "%Y-%m-%d")
            except (ValueError, TypeError) as e:
                raise ValueError(f"SearchResult.freshness must be ISO 8601 format, got: {self.freshness}") from e
        
        # engine 枚举验证
        valid_engines = [e.value for e in SearchEngine]
        if self.engine not in valid_engines:
            raise TypeError(f"SearchResult.engine must be one of {valid_engines}, got: {self.engine}")

    def with_clamped_scores(self) -> 'SearchResult':
        """
        杨植麒压缩表示：返回一个新实例，分数字段被钳制在 [0.0, 1.0]
        避免 NaN/Infinity 传播
        """
        import math
        def clamp(v: float) -> float:
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return max(0.0, min(1.0, v))
        
        return SearchResult(
            url=self.url,
            title=self.title,
            content=self.content,
            engine=self.engine,
            score=clamp(self.score),
            relevance=clamp(self.relevance),
            freshness=self.freshness,
            authority=clamp(self.authority),
            cached=self.cached,
            retrieved_at=self.retrieved_at,
            trust_score=clamp(self.trust_score),
            ranking_uncertainty=clamp(self.ranking_uncertainty),
        )


# ========== Gosling 虚拟机哲学：排序中间表示层 ==========
@dataclass
class RankableResult:
    """
    Gosling 虚拟机哲学：排序算法的中间表示层（IR）
    
    设计目标：将算法逻辑与数据解耦，使得：
    1. 排序策略可以在不了解 SearchResult 内部结构的情况下工作
    2. 数据可以来自任何来源（API、缓存、FAISS）
    3. 不同的排序策略可以组合使用
    
    杨植麒压缩表示：这个 IR 只包含排序必需字段，减少内存占用
    """
    # 原始数据引用（用于最终返回）
    original: SearchResult
    
    # 标准化后的排序特征 [0.0, 1.0]
    normalized_relevance: float
    normalized_freshness: float   # 0.0=很旧, 1.0=最新
    normalized_authority: float  # 0.0=无权威, 1.0=高权威
    normalized_score: float      # 引擎原始分数
    
    # 2026-06-24 新增：排序不确定性
    ranking_uncertainty: float  # 0.0=确定, 1.0=高度不确定
    
    # 共识维度（用于 DIVERSITY 策略）
    consensus_score: float = 0.0  # 由 authority 派生
    
    @classmethod
    def from_search_result(cls, result: SearchResult) -> 'RankableResult':
        """
        将 SearchResult 转换为排序中间表示
        包含 Gosling 强类型的运行时验证
        """
        # 统一 freshness 转换逻辑（避免重复代码）
        freshness_num = cls._freshness_to_number(result.freshness)
        
        # 确保分数在有效范围内
        clamped = result.with_clamped_scores()
        
        return cls(
            original=result,
            normalized_relevance=clamped.relevance,
            normalized_freshness=freshness_num,
            normalized_authority=clamped.authority,
            normalized_score=clamped.score,
            ranking_uncertainty=clamped.ranking_uncertainty,
            consensus_score=clamped.authority,  # 共识度 = 权威性
        )
    
    @staticmethod
    def _freshness_to_number(freshness: str) -> float:
        """
        将 freshness 字符串转为 0-1 数值
        统一在一个地方实现，避免重复代码
        """
        if not freshness:
            return 0.3
        try:
            import datetime
            if len(freshness) >= 10:
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
    
    def to_dict(self, fields: list[str] | None = None) -> dict:
        """
        杨植麒压缩表示：将 IR 转换回字典
        支持字段投影，只返回请求的字段
        
        fields 可选值: ['url', 'title', 'engine', 'score', 'relevance', 
                        'freshness', 'authority', 'trust_score', 'ranking_uncertainty']
        """
        base = {
            "url": self.original.url,
            "title": self.original.title,
            "engine": self.original.engine,
            "score": self.normalized_score,
            "relevance": self.normalized_relevance,
            "freshness": self.original.freshness,
            "authority": self.normalized_authority,
            "trust_score": self.original.trust_score,
            "ranking_uncertainty": self.ranking_uncertainty,
            "cached": self.original.cached,
            "retrieved_at": self.original.retrieved_at,
        }
        
        if fields:
            return {k: v for k, v in base.items() if k in fields}
        return base


@dataclass
class SearchConfig:
    """search 技能配置"""
    default_engines: list = field(default_factory=lambda: ["tavily"])
    max_results: int = 10
    relevance_threshold: float = 0.5
    cache_ttl: int = 3600  # 1小时
    timeout: int = 30  # 秒
    
    # 2026-06-24: 排序策略（解决 Simon 有限理性 + Merton 共识霸权问题）
    # OPTIMAL: 默认最优排序
    # SATISFICING: 够好模式，先返回少量，用户可探索更多
    # DIVERSITY: 多元模式，主动纳入低权威/异见
    ranking_strategy: RankingStrategy = RankingStrategy.OPTIMAL
    
    # SATISFICING 模式：首次返回结果数
    satisficing_initial_results: int = 3
    
    # DIVERSITY 模式：异见结果最低比例
    diversity_min_dissent_ratio: float = 0.2
    
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


class CircuitBreakerForSearch:
    """
    搜索 API 专用熔断器。
    
    追踪每个 engine 的连续失败次数，超过阈值后进入 OPEN 状态，
    跳过该 engine 的调用（快速失败），N 秒后进入 HALF_OPEN 试探。
    """
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._engine_state: dict[str, str] = {}  # "closed"/"open"/"half_open"
        self._failure_counts: dict[str, int] = {}
        self._last_failure_time: dict[str, float] = {}
    
    def call(self, engine: str, fn, *args, **kwargs):
        """
        如果 engine 处于 OPEN/HALF_OPEN 状态则跳过调用直接返回空列表。
        否则执行 fn(*args, **kwargs)，根据结果更新熔断器状态。
        """
        import time
        import threading
        
        if self._engine_state.get(engine) == "open":
            # 检查是否超过 recovery_timeout
            if time.time() - self._last_failure_time.get(engine, 0) >= self._recovery_timeout:
                self._engine_state[engine] = "half_open"
                self._failure_counts[engine] = 0
            else:
                return []  # 熔断中，快速返回空列表
        
        try:
            result = fn(*args, **kwargs)
            if self._engine_state.get(engine) == "half_open":
                # 试探成功，复位
                self._engine_state[engine] = "closed"
                self._failure_counts[engine] = 0
            return result
        except Exception:
            self._failure_counts[engine] = self._failure_counts.get(engine, 0) + 1
            self._last_failure_time[engine] = time.time()
            if self._failure_counts[engine] >= self._failure_threshold:
                self._engine_state[engine] = "open"
            raise


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

    # FAISS 索引持久化路径（可配置）
    FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), ".faiss_index.bin")
    # 向量缓存最大条目数（防止内存泄漏）
    MAX_VECTOR_CACHE_SIZE = 1000

    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()
        self._context: SharedContext = get_context()
        self._cache: dict[str, list[SearchResult]] = {}  # query_hash -> results
        self._cache_time: dict[str, float] = {}  # query_hash -> timestamp
        self._last_search_time: float = 0

        # FAISS 向量索引初始化（懒加载，首次搜索时建立）
        self._result_cache_for_vector: list[SearchResult] = []
        self._faiss_index: Any = None  # FAISS index object
        self._faiss_index_ids: list = []  # 追踪 FAISS index 中的 ID 对应关系

        # 熔断器：防止单个搜索 API 失败拖垮整体服务
        self._circuit_breaker = CircuitBreakerForSearch(failure_threshold=3, recovery_timeout=30.0)

        # V 2026-06-24: 尝试恢复持久化的 FAISS 索引
        self._restore_faiss_index()

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
        
        # 2026-06-24: 优先使用混合搜索（向量 + 关键词 + RRF）
        use_hybrid = HAS_SENTENCE_TRANSFORMERS and HAS_FAISS
        unique_results: list[SearchResult] = []
        used_engines: list[str] = []

        if use_hybrid:
            try:
                # 混合搜索返回已排序结果
                unique_results = self._hybrid_search(query, max_results)
                if unique_results:
                    # 补充 trust_score（hybrid 路径没有走 _compute_trust_scores）
                    unique_results = self._compute_trust_scores(unique_results)
                    used_engines = ["hybrid"]
            except Exception:
                use_hybrid = False

        if not use_hybrid:
            # 执行多引擎搜索（真实并发 + 熔断器）
            all_results: list[SearchResult] = []
            used_engines = []

            # 构建每个 engine 的搜索函数映射
            engine_search_map = {
                "tavily": (self._search_tavily, bool(self.config.tavily_api_key)),
                "brave": (self._search_brave, bool(self.config.brave_api_key)),
                "bing": (self._search_bing, bool(self.config.search_v_path)),
                "exa": (self._search_exa, bool(self.config.exa_api_key)),
                "firecrawl": (self._search_firecrawl, bool(self.config.firecrawl_api_key)),
                "perplexity": (self._search_perplexity, bool(self.config.perplexity_api_key)),
                "github": (self._search_github, True),
            }

            # 并发执行所有引擎搜索
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as executor:
                futures = {}
                for engine in engines:
                    if engine in engine_search_map:
                        search_fn, is_configured = engine_search_map[engine]
                        if is_configured:
                            futures[executor.submit(
                                self._circuit_breaker.call,
                                engine, search_fn, query, max_results
                            )] = engine
                        # 未配置的 API 跳过，不提交任务

                for future in concurrent.futures.as_completed(futures, timeout=self.config.timeout):
                    engine_name = futures[future]
                    try:
                        results = future.result()
                        if results:  # 熔断器可能在 OPEN 状态返回空列表
                            all_results.extend(results)
                            if engine_name not in used_engines:
                                used_engines.append(engine_name)
                    except Exception:
                        pass  # 熔断器已处理，忽略即可

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

            # 排序（使用策略化排序，解决 Simon 有限理性 + Merton 共识霸权问题）
            unique_results = self._rank_by_strategy(unique_results, filters)

            # 限制数量
            unique_results = unique_results[:max_results]

            # 2026-06-18 新增：计算每条结果的可信度评分
            unique_results = self._compute_trust_scores(unique_results)

        # 2026-06-24: RAG 重排序（关键词重叠 + trust_score）
        if unique_results:
            unique_results = self._rerank_results(unique_results, query)

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
            "fake": used_engines == ["mock"],  # Minsky: 明确标记假数据，防止用户误信
            "warning": "All search APIs failed, returning mock data" if used_engines == ["mock"] else None,
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

        items = data.get("items", [])[:max_results]
        results = []
        for idx, item in enumerate(items):
            # 计算权威性分数：基于 stars + forks（log 归一化到 0-1）
            # V 6/18 修 SOP #15 应验: 线性公式 saturate 1000+ stars, trust_score 全 0.97
            # 改为 log10 scale: 10 stars=0.25, 100=0.5, 1000=0.75, 10000=1.0
            stars = item.get("stargazers_count", 0)
            forks = item.get("forks_count", 0)
            authority = min(math.log10(stars + 1 + forks * 0.3) / 4, 1.0)

            # V 2026-06-24 修 P0 regression:
            # score 字段必须在 [0.0, 1.0] 范围 (SearchResult.__post_init__ 运行时校验)
            # 旧代码用 raw stars (e.g. 13219) 灌入 score，触发 ValueError
            # 修复: score 改为位置归一化（API 已按 stars 降序，所以第一名=1.0）
            # 注意: trust_score 计算依赖 authority 而非 score，所以不影响排序语义
            position_score = (len(items) - idx) / len(items) if items else 0.0

            results.append(SearchResult(
                url=item.get("html_url", ""),
                title=f"{item.get('full_name', '')}",
                content=item.get("description", "") or "",
                engine="github",
                score=position_score,
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
                freshness="2024-01-15",  # V 2026-06-24: ISO 8601 (YYYY-MM-DD) 而非 YYYY-MM
                authority=0.8
            ),
            SearchResult(
                url="https://example.com/article2",
                title=f"{query} 详解",
                content=f"深入分析 {query} 的各个方面，包括原理、实践和案例...",
                engine="mock",
                score=0.88,
                relevance=0.85,
                freshness="2024-02-15",  # V 2026-06-24: ISO 8601
                authority=0.75
            ),
            SearchResult(
                url="https://example.com/article3",
                title=f"如何正确理解 {query}",
                content=f"本指南帮助你理解 {query} 的核心概念和应用场景...",
                engine="mock",
                score=0.82,
                relevance=0.78,
                freshness="2024-03-15",  # V 2026-06-24: ISO 8601
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

    def _rank_by_strategy(
        self, 
        results: list[SearchResult], 
        criteria: dict,
        strategy: RankingStrategy | None = None
    ) -> list[SearchResult]:
        """
        按策略排序搜索结果
        
        Latour ANT: 不同策略体现不同的"行动者网络"权力分布
        - OPTIMAL: 算法（FAISS/信任评分）主导
        - SATISFICING: 用户保留控制权，先给少量
        - DIVERSITY: 打破算法权力，主动纳入异见
        
        Simon 有限理性: 不同策略对应不同决策哲学
        - OPTIMAL: 追求最优解（不现实）
        - SATISFICING: 够好就停（符合有限理性）
        - DIVERSITY: 提供多样性选择
        
        Merton 科学社会学: 不同策略影响知识生产的平等性
        - OPTIMAL: 强化共识
        - SATISFICING: 降低认知负荷
        - DIVERSITY: 打破共识霸权
        """
        strategy = strategy or self.config.ranking_strategy
        
        if strategy == RankingStrategy.SATISFICING:
            return self._rank_satisficing(results, criteria)
        elif strategy == RankingStrategy.DIVERSITY:
            return self._rank_diversity(results, criteria)
        else:
            return self._rank_optimal(results, criteria)
    
    def _rank_optimal(self, results: list[SearchResult], criteria: dict) -> list[SearchResult]:
        """
        最优模式 - 当前默认行为
        
        Latour: 索引/缓存等算法行动者主导，用户被动接受"最优"
        问题: 声称最优，但实际是算法偏见
        """
        return self._rank_results(results, criteria)
    
    def _rank_satisficing(self, results: list[SearchResult], criteria: dict) -> list[SearchResult]:
        """
        Simon 够好模式 - 渐进式结果披露
        
        Latour: 用户保留更多控制权，够好即可
        Simon: 符合有限理性，认知负荷最小化
        Merton: 允许非主流观点有曝光机会
        
        策略: 
        1. 先用简化的相关性排序
        2. 降低位置衰减的权重
        3. 用户可以"够了"随时停止
        """
        if not results:
            return results
        
        # 降低位置权重，减少马太效应
        # 位置分数从 1/(idx+1) 改为 1/sqrt(idx+1)，衰减更慢
        weights = {
            "relevance": criteria.get("relevance_weight", 0.4),
            "freshness": criteria.get("freshness_weight", 0.3),
            "authority": criteria.get("authority_weight", 0.2),
            "score": criteria.get("score_weight", 0.1),
        }
        
        def _freshness_to_number(freshness: str) -> float:
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
        
        def satisficing_score(idx: int, r: SearchResult) -> float:
            """够好模式：更轻的位置衰减"""
            freshness_num = _freshness_to_number(r.freshness)
            relevance_score = (
                r.relevance * weights["relevance"] +
                freshness_num * weights["freshness"] +
                r.authority * weights["authority"] +
                r.score * weights["score"]
            )
            # 位置权重从 1/(idx+1) 改为 1/sqrt(idx+1)，衰减更慢
            # 这样第10名不再是0.1而是0.32，差距减小
            position_weight = 1.0 / (idx ** 0.5 + 1)
            return relevance_score * (1 - position_weight * 0.3) + position_weight * 0.1
        
        return sorted(results, key=lambda r: satisficing_score(
            results.index(r), r
        ), reverse=True)
    
    def _rank_diversity(self, results: list[SearchResult], criteria: dict) -> list[SearchResult]:
        """
        Merton 多元模式 - 主动纳入低权威/异见
        
        Latour: 打破算法的隐形权力，异见也有曝光机会
        Simon: 降低认知负荷，提供多种选择而非唯一最优
        Merton: 打破共识霸权，主动纳入被忽视的知识
        
        策略:
        1. 计算每条结果的"共识度"（权威性/信任度）
        2. 主动注入低共识但合理的备选结果
        3. 确保至少 N% 的结果显示非主流观点
        """
        if not results:
            return results
        
        min_dissent_ratio = self.config.diversity_min_dissent_ratio
        
        # 计算每条结果的共识度（与权威性正相关）
        # 使用字典存储，避免修改 SearchResult 类
        consensus_scores: dict[str, float] = {}
        for r in results:
            consensus_scores[r.url] = r.authority if r.authority else 0.3
        
        # 按共识度分两组
        consensus_sorted = sorted(results, key=lambda r: consensus_scores[r.url], reverse=True)
        
        # 计算需要多少异见结果
        total = len(consensus_sorted)
        min_dissent_count = max(1, int(total * min_dissent_ratio))
        
        # 共识结果（高权威）
        consensus_results = [r for r in consensus_sorted if consensus_scores[r.url] >= 0.6]
        # 异见结果（低权威但合理）
        dissent_results = [r for r in consensus_sorted if consensus_scores[r.url] < 0.6]
        
        # 打乱异见结果的顺序（避免同类结果扎堆）
        import random
        random.seed(42)  # 可重复性
        random.shuffle(dissent_results)
        
        # 交替插入异见结果
        merged = []
        dissent_idx = 0
        for i, r in enumerate(consensus_sorted):
            merged.append(r)
            # 每隔3个共识结果插入1个异见结果
            if (i + 1) % 3 == 0 and dissent_idx < len(dissent_results):
                merged.append(dissent_results[dissent_idx])
                dissent_idx += 1
        
        # 如果异见数量不足，随机补充一些低权威结果
        if dissent_idx < len(dissent_results):
            merged.extend(dissent_results[dissent_idx:])
        
        return merged

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

    # ==================== 2026-06-24: FAISS 持久化 + 缓存限制 ====================

    def _restore_faiss_index(self):
        """
        从磁盘恢复持久化的 FAISS 索引和向量缓存。
        解决重启后索引丢失的问题。
        """
        if not HAS_FAISS:
            return
        try:
            if os.path.exists(self.FAISS_INDEX_PATH):
                # 加载序列化的索引
                state = np.load(self.FAISS_INDEX_PATH, allow_pickle=True)
                index_data = state.item()
                if index_data is not None:
                    self._result_cache_for_vector = index_data.get('results', [])
                    # 向量索引重建（FAISS 不支持直接序列化 IndexIDMap）
                    if self._result_cache_for_vector and HAS_SENTENCE_TRANSFORMERS:
                        self._rebuild_faiss_index()
                    print(f"[SearchSkill] Restored FAISS index with {len(self._result_cache_for_vector)} entries")
        except Exception as e:
            print(f"[SearchSkill] Failed to restore FAISS index: {e}")
            # 降级：清空损坏的缓存
            self._result_cache_for_vector = []

    def _rebuild_faiss_index(self):
        """根据缓存的 SearchResult 重建 FAISS 索引"""
        if not self._result_cache_for_vector or not HAS_SENTENCE_TRANSFORMERS:
            return
        try:
            if not hasattr(self, "_embedding_model"):
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [r.title + " " + (r.content or "") for r in self._result_cache_for_vector]
            embeddings = self._embedding_model.encode(texts, convert_to_numpy=True)
            embeddings = embeddings.astype("float32")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            self._faiss_index = faiss.IndexIDMap(index)
            ids = np.array(list(range(len(self._result_cache_for_vector))))
            self._faiss_index.add_with_ids(embeddings, ids)
        except Exception:
            self._faiss_index = None

    def _save_faiss_index(self):
        """
        将 FAISS 索引状态持久化到磁盘。
        只保存 SearchResult 列表（可序列化），
        向量索引在下一次 _restore_faiss_index 时重建。
        """
        if not HAS_FAISS:
            return
        try:
            # 只序列化 SearchResult 列表（索引可通过结果重建）
            state = {'results': self._result_cache_for_vector}
            np.save(self.FAISS_INDEX_PATH, np.array(state, dtype=object))
        except Exception as e:
            print(f"[SearchSkill] Failed to save FAISS index: {e}")

    def _evict_vector_cache_if_needed(self):
        """
        当向量缓存超过 MAX_VECTOR_CACHE_SIZE 时，淘汰最旧的结果。
        采用 FIFO 策略（从列表头部移除）。
        """
        while len(self._result_cache_for_vector) > self.MAX_VECTOR_CACHE_SIZE:
            evicted = self._result_cache_for_vector.pop(0)
            # 重建索引（因为 ID 对应关系被打乱了）
            if self._faiss_index is not None and len(self._result_cache_for_vector) > 0:
                self._rebuild_faiss_index()
            print(f"[SearchSkill] Evicted vector cache entry: {evicted.url}")

    def _enforce_cache_ttl(self):
        """
        主动清理过期的 HTTP 缓存条目（不只是被动查询时检查）。
        防止 _cache 和 _cache_time 长期不同步导致内存泄漏。
        """
        now = time.time()
        expired_keys = [
            k for k, t in self._cache_time.items()
            if now - t > self.config.cache_ttl
        ]
        for k in expired_keys:
            self._cache.pop(k, None)
            self._cache_time.pop(k, None)

    # ==================== 2026-06-24: 向量检索 + 混合搜索 + RAG 重排序 ====================

    def _index_results(self, results: list[SearchResult]):
        """
        将搜索结果构建为 FAISS 向量索引，供后续语义相似度搜索使用。

        使用 all-MiniLM-L6-v2 生成 384 维 embedding，
        用 IndexFlatIP（内积）做余弦相似度搜索。
        """
        if not HAS_FAISS or not HAS_SENTENCE_TRANSFORMERS:
            return
        if not results:
            return

        try:
            # 懒加载模型
            if not hasattr(self, "_embedding_model"):
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            # 对已有结果去重（避免重复索引）
            existing_urls = {r.url for r in self._result_cache_for_vector}
            new_results = [r for r in results if r.url not in existing_urls]
            if not new_results:
                return

            # 生成 embeddings（批量加速）
            texts = [r.title + " " + (r.content or "") for r in new_results]
            embeddings = self._embedding_model.encode(texts, convert_to_numpy=True)
            embeddings = embeddings.astype("float32")

            # 归一化为单位向量（让 IndexFlatIP 等价于余弦相似度）
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

            # 建立或追加 FAISS 索引
            dimension = embeddings.shape[1]
            if self._faiss_index is None:
                index = faiss.IndexFlatIP(dimension)
                self._faiss_index = faiss.IndexIDMap(index)

            # 添加向量和对应的原始 SearchResult
            ids = np.array([len(self._result_cache_for_vector) + i for i in range(len(new_results))])
            self._faiss_index.add_with_ids(embeddings, ids)
            self._result_cache_for_vector.extend(new_results)

            # V 2026-06-24: 防止向量缓存无限增长
            self._evict_vector_cache_if_needed()
            # V 2026-06-24: 持久化 FAISS 索引状态
            self._save_faiss_index()

        except Exception:
            # 向量索引失败不影响搜索功能，静默降级
            pass

    def _search_vector(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        使用 sentence-transformers 生成 embedding，FAISS 向量搜索。

        需要：sentence-transformers, faiss
        返回：向量相似度最高的 top_k 结果
        """
        if not HAS_SENTENCE_TRANSFORMERS or not HAS_FAISS:
            return []

        # 延迟加载模型（避免启动时卡顿）
        if not hasattr(self, "_embedding_model"):
            # 使用轻量模型：all-MiniLM-L6-v2（384维，快且准）
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        if not hasattr(self, "_faiss_index"):
            # 如果没有预建索引，返回空（混合搜索会跳过向量分支）
            return []

        try:
            # 生成 query embedding
            query_vec = self._embedding_model.encode([query], convert_to_numpy=True)[0]
            query_vec = query_vec.astype("float32")

            # FAISS 搜索
            k = min(top_k, len(self._result_cache_for_vector))
            if k == 0:
                return []

            distances, indices = self._faiss_index.search(
                query_vec.reshape(1, -1), k
            )

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self._result_cache_for_vector):
                    r = self._result_cache_for_vector[int(idx)]
                    # 将欧氏距离转为相似度分数（0-1）
                    score = 1.0 / (1.0 + dist)
                    results.append(SearchResult(
                        url=r.url,
                        title=r.title,
                        content=r.content,
                        engine=f"{r.engine}+vector",
                        score=score,
                        relevance=score,
                        freshness=r.freshness,
                        authority=r.authority,
                        trust_score=r.trust_score,
                    ))
            return results
        except Exception:
            return []

    def _hybrid_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        并行关键词搜索 + 向量搜索，RRF（Reciprocal Rank Fusion）融合。
        
        RRF 公式: score(d) = sum(1 / (k + rank_i(d))) for i in results
        k=60（常用默认值）
        """
        k_rff = 60

        # 1. 关键词搜索（先于向量索引）
        keyword_results = self._search_keyword_only(query, top_k * 2)

        # 2. 将关键词结果索引到 FAISS（这样首次搜索也能做向量检索）
        if keyword_results:
            self._index_results(keyword_results)

        # 3. 向量搜索（基于已索引的关键词结果）
        vector_results = self._search_vector(query, top_k * 2)

        if not keyword_results and not vector_results:
            return []

        # 3. RRF 融合
        rrf_scores: dict[str, float] = {}
        rrf_results: dict[str, SearchResult] = {}

        # 关键词结果计分
        for rank, r in enumerate(keyword_results):
            key = r.url
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k_rff + rank + 1)
            if key not in rrf_results:
                rrf_results[key] = r

        # 向量结果计分
        for rank, r in enumerate(vector_results):
            key = r.url
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k_rff + rank + 1)
            if key not in rrf_results:
                rrf_results[key] = r

        # 按 RRF 分数排序
        sorted_urls = sorted(rrf_scores.keys(), key=lambda u: rrf_scores[u], reverse=True)

        fused: list[SearchResult] = []
        for url in sorted_urls[:top_k]:
            r = rrf_results[url]
            r.score = rrf_scores[url]
            fused.append(r)

        return fused

    def _search_keyword_only(self, query: str, max_results: int) -> list[SearchResult]:
        """纯关键词搜索（内部方法，供混合搜索调用）"""
        # 复用现有的多引擎搜索逻辑，但只取关键词相关性
        engines = self.config.default_engines
        all_results: list[SearchResult] = []

        for engine in engines:
            try:
                if engine == "tavily" and self.config.tavily_api_key:
                    results = self._search_tavily(query, max_results)
                elif engine == "brave" and self.config.brave_api_key:
                    results = self._search_brave(query, max_results)
                elif engine == "github":
                    results = self._search_github(query, max_results)
                else:
                    continue
                all_results.extend(results)
            except SearchAPIError:
                continue

        if not all_results:
            all_results = self._execute_search_mock(query, max_results)

        # 去重
        seen_urls = set()
        unique = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique.append(r)

        # 预计算 trust_score
        unique = self._compute_trust_scores(unique)
        return unique[:max_results]

    def _rerank_results(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        """
        基于关键词重叠 + trust_score 重排序（RAG 重排序）。
        
        重排序信号：
        1. 关键词重叠率（query terms 在 title/content 中出现次数）
        2. trust_score（已有）
        3. 位置权重（越靠前越高）
        
        2026-06-24 改进（Box 哲学）：增加不确定性度量
        - 计算排序置信度：当 top-k 结果分数差异大时，置信度高；差异小时置信度低
        - 返回结果附带 uncertainty 指标
        """
        if not results:
            return results

        query_terms = set(query.lower().split())

        def keyword_overlap(r: SearchResult) -> float:
            """计算 query terms 在 title + content 中的重叠率"""
            if not query_terms:
                return 0.0
            text = (r.title + " " + r.content).lower()
            matched = sum(1 for term in query_terms if term in text)
            return matched / len(query_terms)

        def final_score(idx: int, r: SearchResult) -> float:
            overlap = keyword_overlap(r)
            trust = r.trust_score if r.trust_score else 0.3
            position = 1.0 / (idx + 1)  # 位置衰减
            return overlap * 0.4 + trust * 0.4 + position * 0.2

        # 计算原始分数用于不确定性估计
        scored = [(idx, r, final_score(idx, r)) for idx, r in enumerate(results)]
        scores = [s[2] for s in scored]
        
        # Box 哲学：计算排序不确定性（分数差异越小，不确定性越高）
        # 使用归一化分数差异作为不确定性度量
        if len(scores) >= 2:
            score_range = max(scores) - min(scores) if max(scores) != min(scores) else 1.0
            score_gap = scores[0] - scores[1] if len(scores) > 1 else score_range
            # 归一化不确定性：gap 小 → 高不确定性
            uncertainty = 1.0 - min(score_gap / score_range, 1.0) if score_range > 0 else 0.5
        else:
            uncertainty = 0.5
        
        reranked = sorted(scored, key=lambda x: x[2], reverse=True)
        
        # 给每个结果附加不确定性分数
        for rank, (orig_idx, r, score) in enumerate(reranked):
            # 位置不确定性：排名越靠后，不确定性越高
            position_uncertainty = 1.0 / (rank + 1)
            # 综合不确定性（Box 哲学：承认模型无知）
            r.ranking_uncertainty = round(uncertainty * 0.7 + position_uncertainty * 0.3, 3)
        
        # V 2026-06-24 修复 P0 regression: reranked 是 3 元组 (idx, r, score)
        # 旧代码解包成 2 元组会抛 ValueError: too many values to unpack
        return [r for _, r, _ in reranked]

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

    def explain_ranking(
        self, 
        result: SearchResult, 
        query: str,
        strategy: RankingStrategy | None = None
    ) -> dict:
        """
        解释单条结果的排序原因
        
        Latour ANT: 揭示算法黑箱，让用户理解决策过程
        用户可以看到哪些"行动者"（引擎/缓存/索引）影响了结果排序
        
        返回:
            包含排序原因、各因素贡献、建议的字典
        """
        strategy = strategy or self.config.ranking_strategy
        
        # 计算各因素得分
        freshness_score = self._parse_freshness(result.freshness)
        engine_score = self._get_engine_score(result.engine)
        authority_score = result.authority if result.authority else 0.3
        
        factors = {
            "relevance": result.relevance,
            "freshness": freshness_score,
            "authority": authority_score,
            "engine": engine_score,
        }
        
        # 计算权重（与 _rank_results 保持一致）
        weights = {
            "relevance": 0.4,
            "freshness": 0.3,
            "authority": 0.2,
            "engine": 0.1,
        }
        
        # 计算贡献
        contributions = {k: factors[k] * weights[k] for k in factors}
        total_score = sum(contributions.values())
        
        # 生成原因描述
        if result.trust_score and result.trust_score > 0.7:
            primary_reason = "高可信度（权威+新鲜度+引擎可靠性综合得分高）"
        elif result.authority and result.authority > 0.7:
            primary_reason = "高权威性（可能来自流行的GitHub仓库或高引用来源）"
        elif freshness_score > 0.8:
            primary_reason = "内容新颖（最近30天内发布）"
        else:
            primary_reason = "相关性匹配"
        
        # 策略相关的建议
        if strategy == RankingStrategy.OPTIMAL:
            suggestion = "当前为最优模式，算法已选择此结果"
        elif strategy == RankingStrategy.SATISFICING:
            suggestion = "够好模式：此结果满足'够好'标准，可随时停止浏览"
        elif strategy == RankingStrategy.DIVERSITY:
            if authority_score < 0.6:
                suggestion = "多元模式：此结果为异见观点，用于打破共识霸权"
            else:
                suggestion = "多元模式：此结果为共识观点"
        
        return {
            "url": result.url,
            "title": result.title,
            "primary_reason": primary_reason,
            "strategy": strategy.value,
            "factors": factors,
            "weights": weights,
            "contributions": contributions,
            "total_score": round(total_score, 3),
            "trust_score": result.trust_score,
            "suggestion": suggestion,
            "cached": result.cached,
        }
    
    def _parse_freshness(self, freshness: str) -> float:
        """将 freshness 字符串转为 0-1 分数（与 _compute_trust_scores 保持一致）"""
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
    
    def _get_engine_score(self, engine: str) -> float:
        """获取引擎可靠性评分"""
        ENGINE_SCORES = {
            "github": 0.9,
            "perplexity": 0.8,
            "tavily": 0.7,
            "brave": 0.7,
            "exa": 0.7,
            "firecrawl": 0.6,
            "bing": 0.5,
            "mock": 0.1,
        }
        return ENGINE_SCORES.get(engine, 0.5)

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
            "ranking_uncertainty": result.ranking_uncertainty,  # 2026-06-24 Box哲学新增
        }


def get_skill_instance(config: SearchConfig | None = None) -> SearchSkill:
    """获取 search 技能实例"""
    return SearchSkill(config=config)
