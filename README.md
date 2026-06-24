# AgentSearch

多引擎搜索技能，为 Agent 提供智能信息检索能力。

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

## Trust Score 算法

```python
# 信任分 = 加权权威分(对数) + 新鲜度分 + 引擎分
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
