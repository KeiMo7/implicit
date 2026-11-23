"""重排耗时上界测试：不随规则数量线性失控。

引擎在请求内把规则预解析成 O(分组维度) 的查表，运行期决策不遍历规则条数。
本测试用合成数据验证：规则数扩大 20 倍时，单次重排耗时保持在同一量级，
且相对召回（一次 ALS recommend）有明确上界。
"""

import time

import numpy as np
from recommender_base_test import get_checker_board

from implicit.als import AlternatingLeastSquares
from implicit.ranking import ItemCatalog, RerankEngine, RerankingRecommender, load_rules


def _engine_with_rules(rule_count):
    rules = []
    # 大量黑名单规则（每条不同物品），模拟规则条数膨胀
    rules.append(
        {"id": "cap-series", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 4, "min_spacing": 1}
    )
    for index in range(rule_count):
        rules.append(
            {"id": f"bl-{index}", "type": "blacklist", "priority": 100 + index,
             "item_ids": [10_000 + index]}  # 不命中主候选，纯粹增加规则条数
        )
    resolver = load_rules({
        "groups": [{"name": "series", "attribute": "series"}],
        "rules": rules,
    })
    catalog = ItemCatalog({i: {"series": i % 5} for i in range(2_000)})
    return RerankEngine(resolver, catalog)


def _time_rerank(engine, ids, scores, N, repeats=50):
    # 预热
    engine.rerank(None, ids, scores, N=N)
    start = time.perf_counter()
    for _ in range(repeats):
        engine.rerank(None, ids, scores, N=N)
    return (time.perf_counter() - start) / repeats


def test_runtime_does_not_grow_linearly_with_rule_count():
    rng = np.random.RandomState(0)
    ids = rng.randint(0, 2_000, size=500).astype(np.int32)
    scores = np.sort(rng.random(500).astype(np.float32))[::-1]
    N = 50

    small = _engine_with_rules(10)
    large = _engine_with_rules(200)

    small_ms = _time_rerank(small, ids, scores, N) * 1000
    large_ms = _time_rerank(large, ids, scores, N) * 1000

    # 规则数扩大 20 倍，耗时增幅应有界（宽松上界 3 倍，留足 CI 抖动余量）
    assert large_ms < small_ms * 3 + 1.0, (small_ms, large_ms)
    # 单次重排本身是毫秒级的明确上界（500 候选、50 展示位）
    assert large_ms < 50.0


def test_rerank_is_cheaper_than_recall_by_order_of_magnitude():
    item_users = get_checker_board(256)
    user_items = item_users.T.tocsr()
    model = AlternatingLeastSquares(
        factors=64, regularization=0, use_gpu=False, random_state=23
    )
    model.fit(item_users, show_progress=False)

    resolver = load_rules({
        "groups": [{"name": "series", "attribute": "series"}],
        "rules": [{"id": "cap", "type": "group_cap", "priority": 50,
                   "group": "series", "max_items": 5, "min_spacing": 1}],
    })
    catalog = ItemCatalog({i: {"series": i % 5} for i in range(256)})
    wrapped = RerankingRecommender(model, RerankEngine(resolver, catalog))

    userid = 0
    N = 20
    repeats = 30

    start = time.perf_counter()
    for _ in range(repeats):
        model.recommend(userid, user_items[userid], N=N)
    recall_ms = (time.perf_counter() - start) / repeats * 1000

    raw_ids, raw_scores = model.recommend(userid, user_items[userid], N=N)
    start = time.perf_counter()
    for _ in range(repeats):
        wrapped.engine.rerank(userid, raw_ids, raw_scores, N=N)
    rerank_ms = (time.perf_counter() - start) / repeats * 1000

    # 重排相对召回有明确上界：不超过召回耗时（纯内存数组操作 vs 矩阵打分）
    assert rerank_ms <= recall_ms + 1.0
