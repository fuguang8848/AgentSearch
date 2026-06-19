# AgentSearch

多引擎搜索技能，为 Agent 提供智能信息检索能力。

> **V 6/19 18:38 README 修真** (SOP #34 跨仓 L1 对比, 修真 AgentSearch):
> - 失实 1: 7 引擎清单 — 实际 SearchEngine enum 8 引擎 (TAVILY/BRAVE/EXA/FIRECRAWL/PERPLEXITY/MOCK/BING/GITHUB)
> - 失实 2: Exa/Perplexity/Firecrawl 标"未实现" — 实际 8 引擎全实现 (_search_xxx 方法)
> - 失实 3: README 缺 GITHUB 引擎 (V 6/18 新增, 无需认证, REST API)
> - 失实 4: README 缺 trust_score log scale 算法 (V 6/18 SOP #15 应验修)
> - 失实 5: `AgentSearchSkill` — 实际是 `SearchSkill` (顶层)
> - 失实 6: SearchSkill 暴露方法是 `execute(action, params)` / `query(capability, context)` / `notify(event, data)`

## 特性

- **多引擎支持**：8 引擎 (Tavily / Brave / Exa / Perplexity / Firecrawl / Bing / GitHub / Mock)
- **结果路由**：根据调用来源决定输出格式
- **智能去重**：按 URL 自动去重
- **缓存机制**：避免重复搜索
- **优雅降级**：API key 缺失时自动 fallback 到 mock 或下一个引擎

## 引擎清单 (V 6/19 18:38 L1 验证, 8 引擎全实现)

| 引擎 | API Key | 状态 | 实现 | 备注 |
|---|---|---|---|---|
| **Tavily** | `tavily_api_key` | ✅ 可用 | `_search_tavily` line 330 | [docs.tavily.com](https://docs.tavily.com/) |
| **Brave** | `brave_api_key` | ✅ 可用 | `_search_brave` line 392 | [brave.com/search/api](https://api.search.brave.com/) |
| **Exa** | `exa_api_key` | ✅ 可用 | `_search_exa` line 452 | [exa.ai](https://exa.ai) |
| **Perplexity** | `perplexity_api_key` | ✅ 可用 | `_search_perplexity` line 566 | Sonar 模型 |
| **Firecrawl** | `firecrawl_api_key` | ✅ 可用 | `_search_firecrawl` line 511 | `/search` 端点 |
| **Bing** | (无需 key) | ✅ 可用 | `_search_bing` line 722 | 调用 V 端 `search-v.py` (Bing HTML 解析) |
| **GitHub** | (无需 key) | ✅ V 6/18 新增 | `_search_github` line 651 | GitHub REST API 搜索 |
| **Mock** | (无需 key) | ✅ 兜底 | `_execute_search_mock` line 799 | 当所有真实 API 都失败时使用 |

## Trust Score 算法 (V 6/18 修真, SOP #15 #5 应验)

```python
# V 6/18 修: 旧线性公式 saturate 1000+ stars, trust_score 全 0.97
# 改 log10 scale: 10 stars=0.25, 100=0.5, 1000=0.75, 10000=1.0
authority = min(math.log10(stars + 1 + forks * 0.3) / 4, 1.0)
trust_score = w_authority * authority + w_freshness * freshness_score + w_engine * engine_score
```

## 安装

```bash
git clone https://github.com/YintaTriss/AgentSearch.git
cd AgentSearch
pip install -e .

# 可选: Bing 引擎依赖 (V 端 search-v.py 通过 httpx + lxml)
pip install -e .[bing]
```

> **PEP 668 提示**：Ubuntu 24.04 等系统需要 `--break-system-packages` 或 venv。

## 使用 (V 6/19 18:38 L1 验证)

```python
from agent_search import SearchSkill

skill = SearchSkill()
# SearchSkill 暴露方法: execute(action, params) / query(capability, context) / notify(event, data)
result = skill.execute(
    action='search',
    params={'query': 'AgentSearch 8 引擎', 'engines': ['tavily', 'brave', 'github'], 'top_k': 10},
)
```

## 文档

详见 `agent_search/` 包内每个模块的 docstring.
