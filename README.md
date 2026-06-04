# AgentSearch

多引擎搜索技能，为 Agent 提供智能信息检索能力。

## 特性

- **多引擎支持**：Tavily、Brave Search、Exa、Perplexity、Firecrawl、**Bing**
- **结果路由**：根据调用来源决定输出格式
- **智能去重**：按 URL 自动去重
- **缓存机制**：避免重复搜索
- **优雅降级**：API key 缺失时自动 fallback 到 mock 或下一个引擎

## 引擎清单

| 引擎 | API Key | 状态 | 备注 |
|---|---|---|---|
| **Tavily** | `tavily_api_key` | ✅ 可用 | [docs.tavily.com](https://docs.tavily.com/) |
| **Brave Search** | `brave_api_key` | ✅ 可用 | [brave.com/search/api](https://api.search.brave.com/app/documentation/web-search/get-started) |
| **Exa** | (未实现) | ⏸ | 占位，需要 API key |
| **Perplexity** | (未实现) | ⏸ | 占位，需要 API key |
| **Firecrawl** | (未实现) | ⏸ | 占位，需要 API key |
| **Bing** | (无需 key) | ✅ V 端集成 | 调用 V 端 `search-v.py` (Bing HTML 解析) |
| **Mock** | (无需 key) | ✅ 兜底 | 当所有真实 API 都失败时使用 |

## 安装

```bash
git clone https://github.com/YintaTriss/AgentSearch.git
cd AgentSearch
pip install -e .

# 可选：Bing 引擎依赖 (httpx + lxml)
pip install -e .[bing]
```

> **PEP 668 提示**：Ubuntu 24.04 等系统需要 `--break-system-packages` 或 venv。

## 使用

### 基础用法

```python
from agent_search import SearchSkill, SearchConfig, SearchEngine

# 创建实例（Tavily + Brave）
config = SearchConfig(
    tavily_api_key="tvly-xxx",
    brave_api_key="BSAxxx",
)
search = SearchSkill(config)

# 执行搜索
result = search.execute("search", {
    "query": "OpenClaw skills documentation",
    "engines": ["tavily", "brave"],
    "max_results": 10
})
```

### Bing 引擎（V 端集成）

```python
import os
os.environ["SEARCH_V_PATH"] = "/path/to/search-v.py"  # V 端 search-v.py 路径

from agent_search import SearchSkill, SearchConfig

config = SearchConfig(
    search_v_path=os.environ["SEARCH_V_PATH"],
)
search = SearchSkill(config)

result = search.execute("search", {
    "query": "OpenClaw",
    "engines": ["bing"],  # 显式指定 bing 引擎
    "max_results": 5,
})
```

**Bing 引擎优势**：无需 API key，依赖 V 端成熟的 `search-v.py`（Bing HTML 解析 + 防反爬 + 自动重试）。

### 混合多引擎

```python
result = search.execute("search", {
    "query": "大模型 Agent 框架对比",
    "engines": ["tavily", "bing"],  # Tavily 有 key 才用，没 key 跳到 bing
    "max_results": 10,
})
# engines: ['tavily', 'bing']  (实际用了哪些)
# count: 15                    (去重后的总数)
```

## 架构

```
AgentSearch/
├── agent_search/      # 主模块
│   ├── __init__.py
│   └── skill.py        # 核心技能实现（含 _search_tavily / _search_brave / _search_bing / _execute_search_mock）
├── tests/              # 单元测试
│   └── test_smoke.py   # 5/5 check 验证
├── SKILL.md            # OpenClaw 技能入口
├── pyproject.toml      # 包配置（4 optional-deps）
├── .gitignore
└── README.md           # 本文件
```

## 作为 AgentSymphony 的一部分

AgentSearch 是 [AgentSymphony](https://github.com/YintaTriss/AgentSymphony) 技能交响乐的子技能。可选 `pip install -e .[symphony]`。

## 作为 AgentTeam 的一部分

AgentSearch 可被 AgentTeam 调度。可选 `pip install -e .[team]`。

## 独立使用

`agent_search.skill` 模块**独立可装**（顶部 `agent_symphony` 依赖为可选）：
- 装了 AgentSymphony：用真 SharedContext
- 没装：用本地 mock fallback

## License

MIT
