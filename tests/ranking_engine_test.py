"""重排引擎测试：限流、打散、黑名单、强制保留、回填、裁决与确定性。"""

import numpy as np
import pytest

from implicit.ranking import ItemCatalog, RerankEngine, RuleConfigError, load_rules


def _catalog(series_of=None, shop_of=None):
    series_of = series_of or {}
    shop_of = shop_of or {}
    attributes = {}
    for item_id, value in series_of.items():
        attributes.setdefault(item_id, {})["series"] = value
    for item_id, value in shop_of.items():
        attributes.setdefault(item_id, {})["shop"] = value
    return ItemCatalog(attributes)


def _resolver(rules, groups=None):
    return load_rules({"groups": groups or [{"name": "series", "attribute": "series"}],
                       "rules": rules})


def _ids_scores(n=12, start=0):
    ids = np.arange(start, start + n, dtype=np.int32)
    scores = np.linspace(n, 1, n).astype(np.float32)
    return ids, scores


def _positions_by_series(result):
    grouped = {}
    for position in result.positions:
        # 位置记录与最终列表同源
        assert result.ids[position["position"]] == position["item_id"]
        series = position["item_id"] % 3
        grouped.setdefault(series, []).append(position["position"])
    return grouped


# ----------------------------------------------------------------------
# 1. 同系列扎堆 -> 限流与打散后的位置分布
# ----------------------------------------------------------------------
def test_group_cap_limits_series_occupancy():
    # 12 个物品，系列 = item % 3，召回顺序让 series 0 扎堆在最前
    ids = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=np.int32)
    scores = np.linspace(12, 1, 12).astype(np.float32)
    resolver = _resolver([
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 2},
    ])
    engine = RerankEngine(resolver, _catalog({i: i % 3 for i in ids.tolist()}))
    result = engine.rerank(None, ids, scores, N=8)

    counts = {}
    for item_id in result.ids:
        counts[item_id % 3] = counts.get(item_id % 3, 0) + 1
    assert all(count <= 2 for count in counts.values())
    # 被限流的物品有剔除留痕，规则来源可对上
    cap_removed = [r for r in result.removed if r["reason"] == "group_cap_exhausted"]
    assert cap_removed and all(r["rule_id"] == "cap" for r in cap_removed)


def test_spacing_enforces_min_distance():
    ids = np.arange(12, dtype=np.int32)
    scores = np.linspace(12, 1, 12).astype(np.float32)
    resolver = _resolver([
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 4, "min_spacing": 2},
    ])
    engine = RerankEngine(resolver, _catalog({i: i % 3 for i in range(12)}))
    result = engine.rerank(None, ids, scores, N=10)

    grouped = _positions_by_series(result)
    for positions in grouped.values():
        for earlier, later in zip(positions, positions[1:]):
            # 同组相邻条目之间至少间隔 2 个位置（位置差 > 2）
            assert later - earlier > 2, (positions,)


def test_multi_dimensional_groups_series_and_shop():
    ids = np.arange(12, dtype=np.int32)
    scores = np.linspace(12, 1, 12).astype(np.float32)
    resolver = _resolver(
        [
            {"id": "series-cap", "type": "group_cap", "priority": 50,
             "group": "series", "max_items": 3},
            {"id": "shop-cap", "type": "group_cap", "priority": 40,
             "group": "shop", "max_items": 2},
        ],
        groups=[
            {"name": "series", "attribute": "series"},
            {"name": "shop", "attribute": "shop"},
        ],
    )
    catalog = _catalog(
        series_of={i: i % 3 for i in range(12)},
        shop_of={i: i % 2 for i in range(12)},
    )
    engine = RerankEngine(resolver, catalog)
    result = engine.rerank(None, ids, scores, N=10)

    series_counts, shop_counts = {}, {}
    for item_id in result.ids:
        series_counts[item_id % 3] = series_counts.get(item_id % 3, 0) + 1
        shop_counts[item_id % 2] = shop_counts.get(item_id % 2, 0) + 1
    assert all(c <= 3 for c in series_counts.values())
    assert all(c <= 2 for c in shop_counts.values())


def test_ungrouped_items_are_not_throttled():
    ids = np.arange(6, dtype=np.int32)
    scores = np.linspace(6, 1, 6).astype(np.float32)
    resolver = _resolver([
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 1},
    ])
    # 只有物品 0 有系列属性，其余全部 UNGROUPED
    engine = RerankEngine(resolver, _catalog({0: 0}))
    result = engine.rerank(None, ids, scores, N=6)
    assert len(result.ids) == 6  # 无属性物品不受限流影响


# ----------------------------------------------------------------------
# 2. 黑名单、强制保留与回填来源
# ----------------------------------------------------------------------
def test_blacklist_removes_items_and_records_source():
    ids, scores = _ids_scores(6)
    resolver = _resolver([
        {"id": "ops-bl", "type": "blacklist", "priority": 100,
         "item_ids": [2, 4], "source": "ops-ticket-1"},
    ])
    engine = RerankEngine(resolver, _catalog())
    result = engine.rerank(None, ids, scores, N=6)

    assert set(result.ids.tolist()).isdisjoint({2, 4})
    removed = {r["item_id"]: r for r in result.removed}
    assert removed[2]["reason"] == "blacklist"
    assert removed[2]["rule_id"] == "ops-bl"
    assert removed[2]["source"] == "ops-ticket-1"


def test_all_blacklisted_returns_empty_with_reason():
    ids, scores = _ids_scores(3)
    resolver = _resolver([
        {"id": "bl", "type": "blacklist", "priority": 100, "item_ids": [0, 1, 2]},
    ])
    engine = RerankEngine(resolver, _catalog())
    result = engine.rerank(None, ids, scores, N=3)
    assert result.is_empty
    assert len(result.ids) == 0 and len(result.scores) == 0
    assert {r["reason"] for r in result.removed} == {"blacklist"}
    # 回填池里的黑名单物品也绝不放行
    result2 = engine.rerank(
        None, ids, scores, N=3,
        backfill_ids=np.array([1, 2]), backfill_scores=np.zeros(2, np.float32),
    )
    assert result2.is_empty


def test_pin_locks_position_and_keeps_item_despite_cap():
    ids, scores = _ids_scores(8)
    resolver = _resolver([
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 1},
        {"id": "pin-hero", "type": "pin", "priority": 90,
         "item_ids": [5], "positions": [0], "source": "campaign-42"},
    ])
    engine = RerankEngine(resolver, _catalog({i: 0 for i in range(8)}))
    result = engine.rerank(None, ids, scores, N=5)

    assert result.ids[0] == 5
    top = result.positions[0]
    assert top["pinned"] is True
    assert top["rules"] == ["pin-hero", "cap"]
    assert "pinned position" in top["reasons"]


def test_pin_missing_from_recall_is_unfulfilled_not_silent():
    ids, scores = _ids_scores(4)
    resolver = _resolver([
        {"id": "pin", "type": "pin", "priority": 90,
         "item_ids": [99], "positions": [0]},
    ])
    engine = RerankEngine(resolver, _catalog())
    result = engine.rerank(None, ids, scores, N=4)
    unfulfilled = [u for u in result.unfulfilled if u["type"] == "pin_item_not_in_recall"]
    assert unfulfilled and unfulfilled[0]["item_id"] == 99


def test_backfill_fills_empty_slots_with_traceable_source():
    # cap 让主候选只能放 2 个，回填池补足
    ids, scores = _ids_scores(6)
    backfill_ids = np.array([100, 101], dtype=np.int32)
    backfill_scores = np.array([0.5, 0.4], dtype=np.float32)
    resolver = _resolver([
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 2},
    ])
    # 主候选全部同系列，回填物品属于别的系列
    catalog = _catalog(
        series_of={**{i: 0 for i in range(6)}, 100: 1, 101: 2}
    )
    engine = RerankEngine(resolver, catalog)
    result = engine.rerank(
        None, ids, scores, N=4,
        backfill_ids=backfill_ids, backfill_scores=backfill_scores,
    )
    assert set(backfill_ids.tolist()).issubset(result.ids.tolist())
    backfilled_positions = [p for p in result.positions if p["backfilled"]]
    assert len(backfilled_positions) == 2
    assert all(p["backfill_source"] == "recall-ranked-beyond-N"
               for p in backfilled_positions)
    # 回填分数原样透传
    score_map = dict(zip(result.ids.tolist(), result.scores.tolist()))
    assert score_map[100] == pytest.approx(0.5)


def test_blacklisted_item_never_reenters_via_backfill():
    ids, scores = _ids_scores(3)
    resolver = _resolver([
        {"id": "bl", "type": "blacklist", "priority": 100, "item_ids": [7]},
    ])
    engine = RerankEngine(resolver, _catalog())
    result = engine.rerank(
        None, ids, scores, N=5,
        backfill_ids=np.array([7, 8]), backfill_scores=np.zeros(2, np.float32),
    )
    assert 7 not in result.ids
    assert 8 in result.ids


# ----------------------------------------------------------------------
# 3. 规则冲突裁决
# ----------------------------------------------------------------------
def test_blacklist_vs_pin_conflict_adjudicated_by_priority():
    ids, scores = _ids_scores(4)
    resolver = _resolver([
        {"id": "bl", "type": "blacklist", "priority": 100, "item_ids": [1]},
        {"id": "pin", "type": "pin", "priority": 90, "item_ids": [1], "positions": [0]},
    ])
    engine = RerankEngine(resolver, _catalog())
    result = engine.rerank(None, ids, scores, N=4)
    assert 1 not in result.ids  # 黑名单优先级更高
    conflict = next(c for c in result.conflicts if c["type"] == "blacklist_vs_pin")
    assert conflict["winner_rule_id"] == "bl"
    assert conflict["loser_rule_id"] == "pin"

    # 反转优先级：pin 胜出，物品保留并锁定
    resolver2 = _resolver([
        {"id": "bl", "type": "blacklist", "priority": 80, "item_ids": [1]},
        {"id": "pin", "type": "pin", "priority": 95, "item_ids": [1], "positions": [0]},
    ])
    result2 = RerankEngine(resolver2, _catalog()).rerank(None, ids, scores, N=4)
    assert result2.ids[0] == 1
    assert any(c["winner_rule_id"] == "pin" for c in result2.conflicts)


def test_duplicate_group_caps_adjudicated_once():
    ids, scores = _ids_scores(6)
    resolver = _resolver([
        {"id": "cap-loose", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 5},
        {"id": "cap-tight", "type": "group_cap", "priority": 80,
         "group": "series", "max_items": 1},
    ])
    engine = RerankEngine(resolver, _catalog({i: 0 for i in range(6)}))
    result = engine.rerank(None, ids, scores, N=6)
    conflict = next(c for c in result.conflicts if c["type"] == "group_cap_conflict")
    assert conflict["winner_rule_id"] == "cap-tight"
    assert conflict["loser_rule_id"] == "cap-loose"
    # 高优先级的严格上限生效
    assert sum(1 for _ in result.ids) == 1


def test_pin_position_conflict_demotes_lower_priority():
    ids, scores = _ids_scores(6)
    resolver = _resolver([
        {"id": "pin-a", "type": "pin", "priority": 90,
         "item_ids": [0], "positions": [0]},
        {"id": "pin-b", "type": "pin", "priority": 70,
         "item_ids": [1], "positions": [0]},
    ])
    engine = RerankEngine(resolver, _catalog())
    result = engine.rerank(None, ids, scores, N=6)
    assert result.ids[0] == 0
    conflict = next(c for c in result.conflicts if c["type"] == "pin_position_conflict")
    assert conflict["winner_rule_id"] == "pin-a"
    # 被降级的 pin 物品仍按普通候选参与，不会丢失
    assert 1 in result.ids


# ----------------------------------------------------------------------
# 4. 去重、确定性与报告一致性
# ----------------------------------------------------------------------
def test_duplicate_items_in_request_deduped():
    ids = np.array([0, 1, 1, 2, 0], dtype=np.int32)
    scores = np.array([5.0, 4.0, 3.9, 3.0, 2.0], dtype=np.float32)
    resolver = _resolver([])
    engine = RerankEngine(resolver, _catalog())
    result = engine.rerank(None, ids, scores, N=5)
    assert result.ids.tolist() == [0, 1, 2]
    dup_removed = [r for r in result.removed if r["reason"] == "duplicate"]
    assert len(dup_removed) == 2


def test_rerank_is_deterministic_across_repeated_calls():
    ids, scores = _ids_scores(12)
    resolver = _resolver([
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 2, "min_spacing": 1},
        {"id": "bl", "type": "blacklist", "priority": 100, "item_ids": [9]},
        {"id": "pin", "type": "pin", "priority": 90, "item_ids": [3], "positions": [0]},
    ])
    engine = RerankEngine(resolver, _catalog({i: i % 3 for i in range(12)}))
    first = engine.rerank(None, ids, scores, N=8)
    for _ in range(5):
        again = engine.rerank(None, ids, scores, N=8)
        assert again.ids.tolist() == first.ids.tolist()
        assert again.scores.tolist() == first.scores.tolist()
        assert again.positions == first.positions
        assert again.removed == first.removed
        assert again.conflicts == first.conflicts


def test_report_and_final_list_are_same_source():
    ids, scores = _ids_scores(8)
    resolver = _resolver([
        {"id": "bl", "type": "blacklist", "priority": 100, "item_ids": [7]},
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 2},
    ])
    engine = RerankEngine(resolver, _catalog({i: i % 3 for i in range(8)}))
    result = engine.rerank(None, ids, scores, N=6)
    # 每个位置报告都能与最终列表逐条对上
    assert [p["item_id"] for p in result.positions] == result.ids.tolist()
    # 每个被剔除物品都不在最终列表里
    final_set = set(result.ids.tolist())
    for record in result.removed:
        assert record["item_id"] not in final_set
    # 原名次/最终名次字段齐全
    assert all("original_rank" in p and "final_rank" in p for p in result.positions)


def test_scores_are_never_modified():
    ids, scores = _ids_scores(8)
    original = scores.copy()
    resolver = _resolver([
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 2},
    ])
    engine = RerankEngine(resolver, _catalog({i: i % 3 for i in range(8)}))
    result = engine.rerank(None, ids, scores, N=6)
    # 输入数组不被修改
    assert np.array_equal(scores, original)
    # 输出分数都是输入分数的子集，值与 dtype 不变
    assert result.scores.dtype == scores.dtype
    for item_id, score in zip(result.ids, result.scores):
        assert score == original[int(np.flatnonzero(ids == item_id)[0])]


def test_to_dict_is_json_serializable():
    import json

    ids, scores = _ids_scores(5)
    resolver = _resolver([
        {"id": "bl", "type": "blacklist", "priority": 100, "item_ids": [4]},
    ])
    result = RerankEngine(resolver, _catalog()).rerank(None, ids, scores, N=5)
    payload = json.dumps(result.to_dict())
    assert "removed" in payload


# ----------------------------------------------------------------------
# 5. 用户维度规则差异与回落留痕
# ----------------------------------------------------------------------
def test_same_group_has_different_cap_per_user():
    config = {
        "groups": [{"name": "series", "attribute": "series"}],
        "rules": [
            {"id": "global-cap", "type": "group_cap", "priority": 50,
             "group": "series", "max_items": 2},
        ],
        "users": {
            "7": {"rules": [
                {"id": "user7-cap", "type": "group_cap", "priority": 50,
                 "group": "series", "max_items": 1},
            ]},
        },
    }
    resolver = load_rules(config)
    catalog = _catalog({i: 0 for i in range(6)})
    ids, scores = _ids_scores(6)

    global_result = RerankEngine(resolver, catalog).rerank(999, ids, scores, N=6)
    user7_result = RerankEngine(resolver, catalog).rerank(7, ids, scores, N=6)
    assert len(global_result.ids) == 2
    assert len(user7_result.ids) == 1
    # 缺省用户回落留痕
    assert global_result.fallback is not None
    assert user7_result.fallback is None


def test_corrupt_user_ruleset_falls_back_with_record():
    config = {
        "groups": [{"name": "series", "attribute": "series"}],
        "rules": [{"id": "g", "type": "group_cap", "priority": 50,
                   "group": "series", "max_items": 2}],
        "users": {"8": {"rules": []}},  # 空规则集 -> 损坏
    }
    resolver = load_rules(config)
    result = RerankEngine(resolver, _catalog({i: 0 for i in range(6)})).rerank(
        8, *_ids_scores(6), N=6
    )
    assert result.fallback["userid"] == 8
    assert result.fallback["fallback"] == "global"
    assert len(result.ids) == 2  # 用的是全局 cap


def test_invalid_config_raises_at_construction():
    with pytest.raises(RuleConfigError):
        load_rules({"groups": [{"name": "s", "attribute": "s"}],
                    "rules": [{"id": "bad", "type": "group_cap",
                               "priority": 1, "group": "s", "max_items": 0}]})


def test_pin_overrides_group_cap_force_keep_semantics():
    # 强制保留优先于限流：被 pin 的物品即使让该系列占位数达到/超过 cap 也保留；
    # pin 物品计入系列总占位，cap 继续约束其余普通候选（不额外放行）
    attrs = _catalog({i: 0 for i in range(5)})  # 全部同系列
    resolver = _resolver([
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 2},
        {"id": "pin", "type": "pin", "priority": 90,
         "item_ids": [1, 2], "positions": [0, 1]},
    ])
    result = RerankEngine(resolver, attrs).rerank(
        None, np.arange(5, dtype=np.int32),
        np.linspace(5, 1, 5, dtype=np.float32), N=5,
    )
    # 两个 pin 物品强制保留在 0、1 位；它们已占满 cap=2，普通候选不再放行
    assert result.ids[:2].tolist() == [1, 2]
    assert len(result.ids) == 2
    assert all(p["pinned"] for p in result.positions)
    # 未放入的普通候选有 cap 剔除留痕，不会丢失记录
    assert any(x["reason"] == "group_cap_exhausted" for x in result.removed)


def test_pin_can_recover_item_from_backfill_pool():
    # pin 目标不在主候选（前 N）而在回填池时，仍要被捞回锁定
    ids = np.array([101, 102], dtype=np.int32)
    scores = np.array([9.0, 8.0], dtype=np.float32)
    backfill_ids = np.array([201, 301], dtype=np.int32)
    backfill_scores = np.array([7.0, 6.0], dtype=np.float32)
    resolver = _resolver([
        {"id": "pin", "type": "pin", "priority": 90,
         "item_ids": [301], "positions": [0], "source": "campaign"},
    ])
    engine = RerankEngine(resolver, _catalog())
    result = engine.rerank(
        None, ids, scores, N=4,
        backfill_ids=backfill_ids, backfill_scores=backfill_scores,
    )
    assert result.ids[0] == 301
    top = result.positions[0]
    assert top["pinned"] is True and top["backfilled"] is True
    assert top["backfill_source"] == "recall-ranked-beyond-N"


def test_spacing_relaxation_is_recorded_not_silent():
    # 候选多样性不足时放宽间距填满，但必须逐条留痕
    attrs = _catalog({1: 0, 2: 0, 3: 1})  # 只有两个系列
    resolver = _resolver([
        {"id": "cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 3, "min_spacing": 2},
    ])
    engine = RerankEngine(resolver, attrs)
    ids = np.array([1, 2, 3], dtype=np.int32)
    scores = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    result = engine.rerank(None, ids, scores, N=3)
    # 系列 0 的两个物品在候选不足下相邻，放宽标记必须存在
    relaxed = [p for p in result.positions
               if "spacing relaxed to fill slot" in p["reasons"]]
    assert relaxed
    # 限流硬上限仍然满足
    assert sum(1 for i in result.ids if attrs.attributes_of(i)["series"] == 0) <= 3


def test_insufficient_candidates_compacts_and_records():
    # 黑名单剔到只剩 2 个，但请求 N=5
    ids, scores = _ids_scores(5)
    resolver = _resolver([
        {"id": "bl", "type": "blacklist", "priority": 100, "item_ids": [2, 3, 4]},
    ])
    result = RerankEngine(resolver, _catalog()).rerank(None, ids, scores, N=5)
    assert result.ids.tolist() == [0, 1]
    assert any("compacted" in reason
               for p in result.positions for reason in p["reasons"])
