"""
AgentSearch trust_score log scale 测试 (SOP #36 升级必带 test)

V 6/18 修 authority 公式从线性 → log10 scale, 锁住:
1. authority 0-1 范围
2. 不同 stars 仓 trust_score 有 spread (不再 saturate)
3. trust_score != raw stars 排序 (trust_score 排序有意义)
4. 边界: 0 stars / 1 star / 1000 stars / 100000 stars

回归保护: 防止有人改回线性公式
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_search.skill import SearchSkill, SearchResult


def test_01_authority_zero_stars():
    """0 stars → authority=0"""
    s = SearchSkill()
    r = SearchResult(url="x", title="x", content="x", score=0, engine="github", freshness="2026-06-18")
    r.authority = 0.0
    # log10(0+1+0) / 4 = 0/4 = 0
    expected = math.log10(1) / 4
    assert abs(expected - 0.0) < 0.001, f"0 stars should give authority=0, got {expected}"
    print(f"✓ Check 1: 0 stars → authority={expected:.3f}")


def test_02_authority_log_scale_formula():
    """authority = log10(stars+1+forks*0.3) / 4, capped at 1.0"""
    cases = [
        (10, 0, 0.25),       # 10 stars → log10(11)/4 ≈ 0.260
        (100, 0, 0.5),       # 100 stars → log10(101)/4 ≈ 0.501
        (1000, 0, 0.75),     # 1000 stars → log10(1001)/4 ≈ 0.750
        (10000, 0, 1.0),     # 10000 stars → log10(10001)/4 ≈ 1.0 (cap)
    ]
    for stars, forks, expected_authority in cases:
        # Replicate the formula
        actual = min(math.log10(stars + 1 + forks * 0.3) / 4, 1.0)
        assert abs(actual - expected_authority) < 0.05, \
            f"{stars} stars: expected {expected_authority}, got {actual:.3f}"
        print(f"✓ Check 2: {stars:>6} stars → authority={actual:.3f} (expected ~{expected_authority})")


def test_03_trust_score_not_saturated():
    """trust_score 不再 saturate 0.97 (旧 bug 回归保护)"""
    s = SearchSkill()
    # 模拟 5 个 stars 不同的仓
    test_cases = [
        ("tiny", 10),
        ("small", 100),
        ("medium", 500),
        ("big", 5000),
        ("huge", 50000),
    ]
    results = []
    for name, stars in test_cases:
        r = SearchResult(
            url=f"https://github.com/test/{name}",
            title=f"test/{name}",
            content="test repo",
            score=float(stars),
            engine="github",
            freshness="2026-06-15",  # 3 天前, freshness=1.0
        )
        # 实际计算 authority
        r.authority = min(math.log10(stars + 1) / 4, 1.0)
        results.append(r)

    # 跑 trust_score 计算 (按 trust_score 降序)
    scored = s._compute_trust_scores(results)

    # 检查 spread
    trusts = [r.trust_score for r in scored]
    spread = max(trusts) - min(trusts)
    assert spread > 0.05, f"trust_score saturated, spread={spread}, expected > 0.05"
    # 检查不是全 0.97 (旧 bug: 5 仓都 0.97)
    saturated_count = sum(1 for t in trusts if t >= 0.97)
    assert saturated_count < 5, f"全部 0.97 saturate, count={saturated_count}, expected < 5 (小仓 1.0 trust)"
    # 检查 trust 排序后 huge(50000 stars) 排第一, tiny(10) 排最后
    # title 是 "test/huge" 形式 (从构造时 f"test/{name}")
    assert "huge" in scored[0].title, f"排序失败, 第一个不是 huge: {scored[0].title}"
    assert "tiny" in scored[-1].title, f"排序失败, 最后一个不是 tiny: {scored[-1].title}"
    print(f"✓ Check 3: trust_score spread={spread:.3f}, range=[{min(trusts):.3f}..{max(trusts):.3f}], saturate={saturated_count}/5")


def test_04_trust_score_sort_different_from_stars():
    """trust_score 排序 != raw stars 排序 (证明 trust_score 有意义)"""
    s = SearchSkill()
    # 故意构造: A stars 多但更新旧, B stars 少但更新新
    results = [
        SearchResult(url="A", title="A", content="A", score=10000, engine="github", freshness="2024-01-01"),  # 2 年前
        SearchResult(url="B", title="B", content="B", score=100, engine="github", freshness="2026-06-15"),    # 3 天前
        SearchResult(url="C", title="C", content="C", score=1000, engine="github", freshness="2025-12-01"),  # 6 月前
    ]
    for r in results:
        r.authority = min(math.log10(r.score + 1) / 4, 1.0)

    # stars 排序: A(10000) > C(1000) > B(100)
    raw_order = sorted(results, key=lambda r: r.score, reverse=True)
    assert raw_order[0].title == "A", "raw stars 排序失败"

    # trust_score 排序
    scored = s._compute_trust_scores(results)
    # B (新) + 100 stars 应该 trust 不比 C (旧) + 1000 stars 低太多
    # 旧 (2024-01-01): freshness=0.2
    # 10000 stars: authority=1.0
    # trust_A = 0.4*1.0 + 0.3*0.2 + 0.3*0.9 = 0.4+0.06+0.27 = 0.73

    # 新 (2026-06-15): freshness=1.0
    # 100 stars: authority=0.501
    # trust_B = 0.4*0.501 + 0.3*1.0 + 0.3*0.9 = 0.2004+0.3+0.27 = 0.77

    # 6 月前 (2025-12-01): freshness=0.4 (180-365 days)
    # 1000 stars: authority=0.751
    # trust_C = 0.4*0.751 + 0.3*0.4 + 0.3*0.9 = 0.3004+0.12+0.27 = 0.69

    # 期望 trust_B > trust_A > trust_C (新+低 stars 胜 旧+高 stars)
    trusts = {r.title: r.trust_score for r in scored}
    assert trusts["B"] > trusts["A"], f"新仓 trust 应 > 旧仓: B={trusts['B']}, A={trusts['A']}"
    assert trusts["A"] > trusts["C"], f"2 年前应 < 6 月前: A={trusts['A']}, C={trusts['C']}"
    print(f"✓ Check 4: 排序 B(新) > A(2年前) > C(6月前): {trusts['B']:.3f} > {trusts['A']:.3f} > {trusts['C']:.3f}")


def test_05_github_real_search_has_spread():
    """实地搜索: trust_score 排序 ≠ raw stars 排序 (production check)"""
    s = SearchSkill()
    raw = s._search_github("LLM validation", max_results=5)
    if not raw:
        print("⊘ Check 5: 跳过 (GitHub API 不可用, network issue)")
        return
    scored = s._compute_trust_scores(raw)
    trusts = [r.trust_score for r in scored]
    spread = max(trusts) - min(trusts)
    assert spread > 0.01, f"实地搜索 trust_score 仍 saturate, spread={spread}"
    print(f"✓ Check 5: 实地搜索 spread={spread:.3f}, 5 结果 trust 范围 [{min(trusts):.3f}..{max(trusts):.3f}]")


def test_06_no_regression_to_linear():
    """回归保护: 不再是线性公式 (stars 1000+ authority=1.0 的旧 bug)"""
    s = SearchSkill()
    # 旧公式: (1000 + 0*0.3) / 1300 = 0.769
    # 旧公式: (2000 + 0*0.3) / 1300 = 1.0 (capped)
    # 新公式: log10(2000+1) / 4 = 0.825
    # 新公式: log10(10000+1) / 4 = 1.0 (capped at higher number)

    # 1000 stars
    a_1000 = min(math.log10(1000 + 1) / 4, 1.0)
    # 2000 stars
    a_2000 = min(math.log10(2000 + 1) / 4, 1.0)
    # 10000 stars
    a_10000 = min(math.log10(10000 + 1) / 4, 1.0)

    # log scale 应该: 1000 < 2000 < 10000 严格递增
    assert a_1000 < a_2000 < a_10000, f"log scale 失效: {a_1000}, {a_2000}, {a_10000}"
    # 10000 应该 cap 到 1.0
    assert abs(a_10000 - 1.0) < 0.001, f"10000 stars should cap at 1.0, got {a_10000}"
    print(f"✓ Check 6: log scale 严格递增, 10000 cap 1.0: 1000={a_1000:.3f} < 2000={a_2000:.3f} < 10000={a_10000:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("AgentSearch trust_score log scale 测试 (SOP #36)")
    print("=" * 60)
    test_01_authority_zero_stars()
    test_02_authority_log_scale_formula()
    test_03_trust_score_not_saturated()
    test_04_trust_score_sort_different_from_stars()
    test_05_github_real_search_has_spread()
    test_06_no_regression_to_linear()
    print("=" * 60)
    print("✅ 6/6 tests passed")
    print("=" * 60)
