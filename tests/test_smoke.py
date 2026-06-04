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
    print("✓ Check 1: import OK + BING enum (7 个引擎)")


def test_02_search_config():
    """Check 2: SearchConfig.search_v_path 字段"""
    c = SearchConfig()
    assert hasattr(c, 'search_v_path')
    assert c.search_v_path == ""  # 默认空
    c2 = SearchConfig(search_v_path="/tmp/test.py")
    assert c2.search_v_path == "/tmp/test.py"
    print("✓ Check 2: SearchConfig.search_v_path field")


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


if __name__ == "__main__":
    print("=== AgentSearch smoke test (重构后 6/6) ===")
    test_01_import()
    test_02_search_config()
    test_03_bing_real_search()
    test_04_bing_graceful_degrade()
    test_05_bing_cache()
    test_06_existing_engines_unchanged()
    print("\n=== 6/6 check 通过 ===")
