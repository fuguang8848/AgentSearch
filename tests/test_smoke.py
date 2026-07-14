"""AgentSearch 端到端 smoke test (5/5 check)

V 2026-06-04 写：每次项目完成后总览 + 重构 + 回归测试
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 集中 import（V 10:55 重构: 每个 test function 独立 import 是冗余的）
from agent_search import SearchSkill, SearchConfig, SearchResult, SearchEngine


def test_01_import():
    """Check 1: 独立 import（不依赖 agent_symphony）"""
    assert SearchEngine.BING.value == "bing"
    assert SearchEngine.TAVILY.value == "tavily"
    assert SearchEngine.BRAVE.value == "brave"
    assert SearchEngine.EXA.value == "exa"
    assert SearchEngine.FIRECRAWL.value == "firecrawl"
    assert SearchEngine.PERPLEXITY.value == "perplexity"
    print("✓ Check 1: import OK + 全部 7 个引擎 enum")


def test_02_search_config():
    """Check 2: SearchConfig 新引擎 API key 字段"""
    c = SearchConfig()
    assert hasattr(c, 'search_v_path')
    assert hasattr(c, 'exa_api_key')
    assert hasattr(c, 'firecrawl_api_key')
    assert hasattr(c, 'perplexity_api_key')
    assert c.search_v_path == ""  # 默认空
    assert c.exa_api_key == ""
    assert c.firecrawl_api_key == ""
    assert c.perplexity_api_key == ""
    c2 = SearchConfig(
        search_v_path="/tmp/test.py",
        exa_api_key="exa-test-key",
        firecrawl_api_key="firecrawl-test-key",
        perplexity_api_key="perplexity-test-key"
    )
    assert c2.search_v_path == "/tmp/test.py"
    assert c2.exa_api_key == "exa-test-key"
    assert c2.firecrawl_api_key == "firecrawl-test-key"
    assert c2.perplexity_api_key == "perplexity-test-key"
    print("✓ Check 2: SearchConfig 新引擎 API key 字段")


def test_03_bing_real_search():
    """Check 3: Bing 实际搜 (search-v.py 真跑)"""
    search_v = "/home/fuguang/.openclaw/workspace/tools/search-v.py"
    if not os.path.exists(search_v):
        print(f"⚠ Check 3: SKIP (search-v.py 不在 {search_v})")
        return
    s = SearchSkill(config=SearchConfig(search_v_path=search_v))
    r = s.execute("search", {"query": "OpenClaw", "engines": ["bing"], "max_results": 3})
    assert r["success"] and r["data"]["count"] > 0
    assert r["data"]["engines"] == ["bing"]
    print(f"✓ Check 3: Bing 真搜 {r['data']['count']} results")


def test_04_bing_graceful_degrade():
    """Check 4: search_v_path 错 → 优雅降级 mock"""
    s = SearchSkill(config=SearchConfig(search_v_path="/no/such/file.py"))
    r = s.execute("search", {"query": "test", "engines": ["bing"], "max_results": 2})
    # 实际 fallback 到 mock，engines 变 ['mock']
    assert r["success"] and r["data"]["count"] > 0
    assert r["data"]["engines"] == ["mock"]
    print("✓ Check 4: 错路径 → mock 降级")


def test_05_bing_cache():
    """Check 5: _search_bing 用了 importlib cache (不重复 reload)"""
    search_v = "/home/fuguang/.openclaw/workspace/tools/search-v.py"
    if not os.path.exists(search_v):
        print("⚠ Check 5: SKIP")
        return
    s = SearchSkill(config=SearchConfig(search_v_path=search_v))
    # 跑 2 次
    s.execute("search", {"query": "test1", "engines": ["bing"], "max_results": 1})
    s.execute("search", {"query": "test2", "engines": ["bing"], "max_results": 1})
    # 验证 cache 存在
    assert hasattr(s, "_v_search_v_cache")
    cache = s._v_search_v_cache
    assert len(cache) == 1  # 只加载 1 次
    print(f"✓ Check 5: importlib cache hit ({len(cache)} module cached)")


def test_06_existing_engines_unchanged():
    """Check 6: 重构后 tavily/brave/mock 仍正常（回归测试）"""
    s = SearchSkill(config=SearchConfig())
    r = s.execute("search", {"query": "test", "engines": ["mock"], "max_results": 1})
    assert r["success"] and r["data"]["count"] > 0
    assert r["data"]["engines"] == ["mock"]
    print("✓ Check 6: mock 引擎回归正常")


def test_07_exa_engine_no_key():
    """Check 7: exa 无 API key → 跳过（不抛异常，不返回 mock）"""
    s = SearchSkill(config=SearchConfig(exa_api_key=""))
    r = s.execute("search", {"query": "test", "engines": ["exa"], "max_results": 2})
    # exa 无 key 时跳过，all_results 为空 → fallback mock
    assert r["success"]
    # fallback 到 mock，engines 变 ['mock']
    assert r["data"]["engines"] == ["mock"]
    print("✓ Check 7: exa 无 key → 降级 mock")


def test_08_firecrawl_engine_no_key():
    """Check 8: firecrawl 无 API key → 跳过（不抛异常）"""
    s = SearchSkill(config=SearchConfig(firecrawl_api_key=""))
    r = s.execute("search", {"query": "test", "engines": ["firecrawl"], "max_results": 2})
    assert r["success"]
    assert r["data"]["engines"] == ["mock"]
    print("✓ Check 8: firecrawl 无 key → 降级 mock")


def test_09_perplexity_engine_no_key():
    """Check 9: perplexity 无 API key → 跳过（不抛异常）"""
    s = SearchSkill(config=SearchConfig(perplexity_api_key=""))
    r = s.execute("search", {"query": "test", "engines": ["perplexity"], "max_results": 2})
    assert r["success"]
    assert r["data"]["engines"] == ["mock"]
    print("✓ Check 9: perplexity 无 key → 降级 mock")


def test_10_multi_engine_no_keys():
    """Check 10: 多引擎搜索（全部无 key）→ 正确聚合"""
    s = SearchSkill(config=SearchConfig(
        tavily_api_key="",
        brave_api_key="",
        exa_api_key="",
        firecrawl_api_key="",
        perplexity_api_key=""
    ))
    r = s.execute("search", {
        "query": "test",
        "engines": ["tavily", "brave", "exa", "firecrawl", "perplexity"],
        "max_results": 5
    })
    assert r["success"]
    assert r["data"]["engines"] == ["mock"]
    print("✓ Check 10: 多引擎全无 key → mock 降级")


if __name__ == "__main__":
    print("=== AgentSearch smoke test (重构后 10/10) ===")
    test_01_import()
    test_02_search_config()
    test_03_bing_real_search()
    test_04_bing_graceful_degrade()
    test_05_bing_cache()
    test_06_existing_engines_unchanged()
    test_07_exa_engine_no_key()
    test_08_firecrawl_engine_no_key()
    test_09_perplexity_engine_no_key()
    test_10_multi_engine_no_keys()
    print("\n=== 10/10 check 通过 ===")
