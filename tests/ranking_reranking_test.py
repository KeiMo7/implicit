"""RerankingRecommender 包装层测试：接口兼容、开关透传、批量/单用户路径。"""

import numpy as np
import pytest
from recommender_base_test import get_checker_board

from implicit.als import AlternatingLeastSquares
from implicit.ranking import (
    ItemCatalog,
    RerankEngine,
    RerankingRecommender,
    load_rules,
)


@pytest.fixture(scope="module")
def fitted_model():
    item_users = get_checker_board(50)
    model = AlternatingLeastSquares(
        factors=32, regularization=0, use_gpu=False, random_state=23
    )
    model.fit(item_users, show_progress=False)
    return model, item_users.T.tocsr()


def _empty_engine():
    resolver = load_rules({"groups": [], "rules": []})
    return RerankEngine(resolver, ItemCatalog({}))


def test_recommend_signature_and_shapes_unchanged(fitted_model):
    model, user_items = fitted_model
    wrapped = RerankingRecommender(model, _empty_engine())

    # 单用户：一维
    ids, scores = wrapped.recommend(0, user_items[0], N=10)
    assert ids.ndim == 1 and scores.ndim == 1
    assert len(ids) == 10

    # 批量：二维
    userids = np.arange(10)
    batch_ids, batch_scores = wrapped.recommend(userids, user_items[userids], N=10)
    assert batch_ids.shape == (10, 10)
    assert batch_scores.shape == (10, 10)


def test_disabled_rerank_is_identical_to_raw_model(fitted_model):
    model, user_items = fitted_model
    wrapped = RerankingRecommender(model, _empty_engine(), enabled=False)

    raw_ids, raw_scores = model.recommend(0, user_items[0], N=10)
    ids, scores = wrapped.recommend(0, user_items[0], N=10)
    assert np.array_equal(ids, raw_ids)
    assert np.array_equal(scores, raw_scores)

    userids = np.arange(10)
    raw_b_ids, raw_b_scores = model.recommend(userids, user_items[userids], N=10)
    b_ids, b_scores = wrapped.recommend(userids, user_items[userids], N=10)
    assert np.array_equal(b_ids, raw_b_ids)
    assert np.array_equal(b_scores, raw_b_scores)


def test_no_rules_preserves_single_user_output_bit_exact(fitted_model):
    model, user_items = fitted_model
    wrapped = RerankingRecommender(model, _empty_engine(), recall_oversample=1)
    raw_ids, raw_scores = model.recommend(0, user_items[0], N=10)
    ids, scores = wrapped.recommend(0, user_items[0], N=10)
    assert np.array_equal(ids, raw_ids)
    assert np.array_equal(scores, raw_scores)
    assert ids.dtype == raw_ids.dtype and scores.dtype == raw_scores.dtype


def test_existing_filter_semantics_unchanged(fitted_model):
    model, user_items = fitted_model
    wrapped = RerankingRecommender(model, _empty_engine())

    ids, _ = wrapped.recommend(0, user_items[0], N=5, filter_items=[0])
    assert 0 not in ids

    # items 与 filter_items 互斥的校验仍由底层抛出
    with pytest.raises(ValueError):
        wrapped.recommend(0, user_items[0], items=np.arange(5), filter_items=[1])


def test_batch_and_single_user_paths_agree(fitted_model):
    model, user_items = fitted_model
    resolver = load_rules({
        "groups": [{"name": "series", "attribute": "series"}],
        "rules": [{"id": "cap", "type": "group_cap", "priority": 50,
                   "group": "series", "max_items": 3}],
    })
    catalog = ItemCatalog({i: {"series": i % 2} for i in range(50)})
    wrapped = RerankingRecommender(model, RerankEngine(resolver, catalog))

    userids = np.arange(6)
    batch_ids, _ = wrapped.recommend(userids, user_items[userids], N=8)
    reports = wrapped.last_report
    for row, userid in enumerate(userids):
        single_ids, _ = wrapped.recommend(int(userid), user_items[userid], N=8)
        # 批量为固定矩形（不足位补 0），单用户为短列表；按报告里的实际
        # 展示长度取有效前缀比较，不能用 id!=0 过滤（0 是合法物品 id）
        valid_len = len(reports[row].ids)
        assert np.array_equal(batch_ids[row, :valid_len], single_ids)


def test_duplicate_users_in_batch_are_deduped(fitted_model):
    model, user_items = fitted_model
    wrapped = RerankingRecommender(model, _empty_engine())
    userids = np.array([0, 0, 1, 2, 1])
    ids, _ = wrapped.recommend(userids, user_items[userids], N=5)
    assert np.array_equal(ids[0], ids[1])  # 同一用户结果复用
    assert np.array_equal(ids[2], ids[4])
    assert len(wrapped.last_report) == 5


def test_blacklist_applied_through_wrapper(fitted_model):
    model, user_items = fitted_model
    raw_ids, _ = model.recommend(0, user_items[0], N=10)
    target = int(raw_ids[0])
    resolver = load_rules({
        "groups": [],
        "rules": [{"id": "bl", "type": "blacklist", "priority": 100,
                   "item_ids": [target], "source": "ops"}],
    })
    wrapped = RerankingRecommender(model, RerankEngine(resolver, ItemCatalog({})))
    ids, scores = wrapped.recommend(0, user_items[0], N=10)
    assert target not in ids
    report = wrapped.last_report
    assert any(r["item_id"] == target and r["reason"] == "blacklist"
               for r in report.removed)


def test_report_available_via_recommend_with_report(fitted_model):
    model, user_items = fitted_model
    wrapped = RerankingRecommender(model, _empty_engine())
    ids, scores, report = wrapped.recommend_with_report(0, user_items[0], N=5)
    assert len(ids) == len(scores) == 5
    assert [p["item_id"] for p in report.positions] == ids.tolist()


def test_engine_is_replaceable(fitted_model):
    model, user_items = fitted_model
    wrapped = RerankingRecommender(model, _empty_engine())

    replacement = _empty_engine()
    wrapped.engine = replacement  # 独立替换，不触碰模型
    assert wrapped.engine is replacement
    ids, _ = wrapped.recommend(0, user_items[0], N=5)
    assert len(ids) == 5


def test_wrapper_rejects_model_without_recommend():
    with pytest.raises(TypeError):
        RerankingRecommender(object(), _empty_engine())


def test_model_scoring_path_not_touched(fitted_model):
    model, user_items = fitted_model
    before = model.user_factors.copy()
    item_before = model.item_factors.copy()
    wrapped = RerankingRecommender(model, _empty_engine())
    wrapped.recommend(np.arange(5), user_items[np.arange(5)], N=5)
    assert np.array_equal(model.user_factors, before)
    assert np.array_equal(model.item_factors, item_before)
