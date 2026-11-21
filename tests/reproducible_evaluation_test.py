"""Tests for implicit.reproducible_evaluation and the opt-in reproducible split.

These cover the acceptance cases:

* two evaluations with the same seed are bit-for-bit identical;
* sampled negatives never contain a held-out (test) item;
* a user with no evaluable sample is skipped and counted, not averaged as zero;
* too few negatives produce an explicit warning;
* turning the reproducible mode off leaves the historical behaviour untouched.
"""

import json
import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse import random as sparse_random

from implicit import __version__ as implicit_version
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import leave_k_out_split, ranking_metrics_at_k
from implicit.reproducible_evaluation import (
    SPLIT_EXTERNAL,
    SPLIT_LEAVE_K_OUT,
    SPLIT_RANDOM,
    EvaluationPlan,
    build_evaluation_plan,
    compare_results,
    evaluate,
    merge_results,
)


def _dense_ratings(n_users=60, n_items=80, density=0.2, seed=0):
    mat = sparse_random(
        n_users, n_items, density=density, format="csr", random_state=seed, dtype=np.float32
    )
    mat.data = np.ones_like(mat.data)
    return mat


def _fitted_model(train, factors=8, iterations=3, seed=42):
    model = AlternatingLeastSquares(factors=factors, iterations=iterations, random_state=seed)
    model.fit(train, show_progress=False)
    return model


@pytest.fixture(scope="module")
def ratings():
    return _dense_ratings()


@pytest.fixture(scope="module")
def plan(ratings):
    return build_evaluation_plan(ratings, K=1, num_negatives=20, seed=20240101)


@pytest.fixture(scope="module")
def model(plan):
    return _fitted_model(plan.train_user_items)


# ---------------------------------------------------------------- reproducibility
def test_same_seed_two_evaluations_are_bit_identical(model, plan):
    first = evaluate(model, plan, K=10)
    second = evaluate(model, plan, K=10)

    assert first.metrics == second.metrics
    # "bit-for-bit" on the actual doubles, not just equality of the mean.
    for name in first.per_user:
        assert np.array_equal(first.per_user[name], second.per_user[name])
        assert first.per_user[name].tobytes() == second.per_user[name].tobytes()


def test_seed_controls_split_and_negatives(ratings):
    a = build_evaluation_plan(ratings, num_negatives=20, seed=7)
    b = build_evaluation_plan(ratings, num_negatives=20, seed=7)
    c = build_evaluation_plan(ratings, num_negatives=20, seed=8)

    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert np.array_equal(a.test_user_items.indices, b.test_user_items.indices)
    assert np.array_equal(a.negative_indices, b.negative_indices)
    assert not np.array_equal(a.negative_indices, c.negative_indices)


def test_split_does_not_depend_on_global_numpy_state(ratings):
    np.random.seed(1)
    a = build_evaluation_plan(ratings, num_negatives=10, seed=123)
    np.random.seed(999)
    b = build_evaluation_plan(ratings, num_negatives=10, seed=123)
    assert a.fingerprint == b.fingerprint
    assert np.array_equal(a.test_user_items.indices, b.test_user_items.indices)


# ------------------------------------------------------------- negative sampling
def test_negatives_exclude_test_items(plan):
    test = plan.test_user_items
    for user in plan.user_ids:
        user = int(user)
        negatives = set(plan.negatives_for(user).tolist())
        held_out = set(test.indices[test.indptr[user] : test.indptr[user + 1]].tolist())
        assert negatives.isdisjoint(held_out), f"user {user}: test items leaked into negatives"


def test_negatives_exclude_train_items(plan):
    train = plan.train_user_items
    for user in plan.user_ids:
        user = int(user)
        negatives = set(plan.negatives_for(user).tolist())
        seen = set(train.indices[train.indptr[user] : train.indptr[user + 1]].tolist())
        assert negatives.isdisjoint(seen), f"user {user}: train items leaked into negatives"


def test_negatives_are_deterministic_and_per_user(plan):
    rebuilt = build_evaluation_plan(
        plan.train_user_items + plan.test_user_items,
        K=1,
        num_negatives=plan.num_negatives,
        seed=plan.seed,
    )
    for user in plan.user_ids[:10]:
        assert np.array_equal(plan.negatives_for(int(user)), rebuilt.negatives_for(int(user)))


def test_insufficient_negatives_warns(ratings):
    n_users, n_items = 5, 8
    # each user interacts with all but one item, so train+test leaves almost nothing
    dense = np.ones((n_users, n_items), dtype=np.float32)
    for u in range(n_users):
        dense[u, u] = 0.0
    dense = csr_matrix(dense)
    with pytest.warns(RuntimeWarning, match="fewer than .* eligible negatives"):
        plan = build_evaluation_plan(dense, num_negatives=5, seed=0)

    assert plan.summary()["insufficient_negatives_users"] > 0
    assert plan.summary()["negatives_per_user"]["min"] < 5


# ------------------------------------------------------------- skipped users
def test_user_without_test_items_is_counted_not_averaged(ratings):
    train = ratings.tocsr()
    test = csr_matrix(train.shape, dtype=train.dtype)
    # Give only user 0 a held-out item; everyone else has none.
    test = test.tolil()
    test[0, 0] = 1.0
    test = test.tocsr()
    train = train.tolil()
    train[0, 0] = 0.0
    train = train.tocsr()

    plan = EvaluationPlan.from_split(train, test, num_negatives=10, seed=0)
    model = _fitted_model(train)
    result = evaluate(model, plan, K=10)

    assert result.config["skipped_users"]["no_test_items"] == train.shape[0] - 1
    assert result.config["evaluated_users"] == 1
    assert len(result.user_ids) == 1


def test_all_users_skipped_raises(ratings):
    train = ratings.tocsr()
    test = csr_matrix(train.shape, dtype=train.dtype)
    plan = EvaluationPlan.from_split(train, test, num_negatives=10, seed=0)
    model = _fitted_model(train)
    with pytest.raises(ValueError, match="No user has an evaluable sample"):
        evaluate(model, plan, K=10)


def test_user_with_no_negatives_is_skipped_and_counted():
    # One user has interacted with every item (so train union test covers the whole
    # catalogue and no negative is eligible); the other users are normal.
    n_users, n_items = 6, 10
    dense = np.zeros((n_users, n_items), dtype=np.float32)
    dense[0, :] = 1.0
    dense[1:, ::2] = 1.0
    ratings = csr_matrix(dense)

    plan = build_evaluation_plan(ratings, K=1, num_negatives=3, seed=0)
    model = _fitted_model(csr_matrix(dense))

    result = evaluate(model, plan, K=5)
    assert 0 in plan.user_ids
    assert result.config["skipped_users"]["no_negatives"] >= 1
    assert 0 not in set(result.user_ids.tolist())
    assert result.config["evaluated_users"] == len(result.user_ids)
    # skipped users are absent from the per-user arrays, not zeros
    assert len(result.per_user["recall"]) == result.config["evaluated_users"]


# -------------------------------------------------------- batching and sharding
def test_batch_size_does_not_change_metrics(model, plan):
    small = evaluate(model, plan, K=10, batch_size=1)
    large = evaluate(model, plan, K=10, batch_size=1000)
    assert small.metrics == large.metrics
    for name in small.per_user:
        assert np.array_equal(small.per_user[name], large.per_user[name])


def test_sharding_and_merge_match_single_run(model, plan):
    whole = evaluate(model, plan, K=10)
    shards = [evaluate(model, plan, K=10, users=whole.user_ids[i::3]) for i in range(3)]
    merged = merge_results(shards)

    assert merged.metrics == whole.metrics
    for name in whole.per_user:
        assert np.array_equal(merged.user_ids, whole.user_ids)
        assert np.array_equal(merged.per_user[name], whole.per_user[name])


def test_merge_rejects_overlapping_shards(model, plan):
    a = evaluate(model, plan, K=10, users=plan.user_ids[:10])
    b = evaluate(model, plan, K=10, users=plan.user_ids[5:15])
    with pytest.raises(ValueError, match="overlap"):
        merge_results([a, b])


# ------------------------------------------------------------------ comparison
def test_compare_identical_results_is_not_significant(plan):
    model = AlternatingLeastSquares(factors=8, iterations=3, random_state=1)
    model.fit(plan.train_user_items, show_progress=False)
    a = evaluate(model, plan, K=10)
    b = evaluate(model, plan, K=10)
    comparison = compare_results(a, b, metric="recall", num_resamples=200, seed=0)
    assert comparison.delta == 0.0
    assert not comparison.significant


def test_compare_requires_same_plan(ratings):
    plan_a = build_evaluation_plan(ratings, num_negatives=10, seed=1)
    plan_b = build_evaluation_plan(ratings, num_negatives=10, seed=2)
    model = _fitted_model(plan_a.train_user_items)
    a = evaluate(model, plan_a, K=10)
    b = evaluate(model, plan_b, K=10)
    with pytest.raises(ValueError, match="different evaluation plans"):
        compare_results(a, b)


# ------------------------------------------------------------ config recording
def test_config_records_split_sampling_and_metrics(model, plan):
    result = evaluate(model, plan, K=10)
    config = result.config

    assert config["split"] == SPLIT_LEAVE_K_OUT
    assert config["seed"] == plan.seed
    assert config["num_negatives"] == plan.num_negatives
    assert config["test_items_are_negatives"] is False
    assert set(config["metric_definitions"]) >= {"recall", "hit_rate", "ndcg"}
    dataset = config["dataset"]
    assert dataset["n_users"] == plan.train_user_items.shape[0]
    assert dataset["n_items"] == plan.train_user_items.shape[1]
    assert 0.0 < dataset["density"] < 1.0
    assert config["plan_fingerprint"] == plan.fingerprint
    assert config["batch_size"] == 1000
    assert config["library_version"] == implicit_version
    assert config["model"]["class"] == "AlternatingLeastSquares"

    # The definitions must be findable in the saved JSON too.
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["config"]["metric_definitions"] == config["metric_definitions"]


def test_from_split_records_external_protocol(ratings):
    train, test = leave_k_out_split(ratings, K=1, random_state=0, shuffle_random_state=0)
    plan = EvaluationPlan.from_split(
        train,
        test,
        num_negatives=10,
        seed=0,
        split=SPLIT_EXTERNAL,
        split_params={"protocol": "leave_one_out", "random_state": 0},
    )
    assert plan.config()["split"] == SPLIT_EXTERNAL
    assert plan.config()["split_params"]["protocol"] == "leave_one_out"


def test_random_split_is_supported(ratings):
    plan = build_evaluation_plan(ratings, split=SPLIT_RANDOM, train_percentage=0.8, seed=3)
    assert plan.split == SPLIT_RANDOM
    assert plan.config()["split_params"]["train_percentage"] == 0.8
    assert plan.test_user_items.nnz > 0


# ----------------------------------------------------------------- persistence
def test_plan_save_load_roundtrip(plan, tmp_path):
    path = tmp_path / "plan.npz"
    plan.save(path)
    loaded = EvaluationPlan.load(path)
    assert loaded.fingerprint == plan.fingerprint
    for user in plan.user_ids[:5]:
        assert np.array_equal(loaded.negatives_for(int(user)), plan.negatives_for(int(user)))
    assert np.array_equal(loaded.test_user_items.indices, plan.test_user_items.indices)


def test_result_save_is_atomic_and_complete(model, plan, tmp_path):
    result = evaluate(model, plan, K=10)
    path = tmp_path / "result"
    result.save(path)
    assert (tmp_path / "result.npz").exists()
    assert (tmp_path / "result.json").exists()
    # no temp files left behind
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp-" in p]
    assert leftovers == []

    loaded = type(result).load(tmp_path / "result.npz")
    assert loaded.metrics == result.metrics
    assert np.array_equal(loaded.user_ids, result.user_ids)


def test_failed_save_leaves_no_partial_files(model, plan, tmp_path, monkeypatch):
    result = evaluate(model, plan, K=10)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(np, "savez", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        result.save(tmp_path / "result")
    assert os.listdir(tmp_path) == []


def test_failed_plan_save_leaves_no_partial_files(plan, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(np, "savez", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        plan.save(tmp_path / "plan")
    assert os.listdir(tmp_path) == []


def test_failed_evaluation_writes_nothing(model, plan, tmp_path):
    """An exception during evaluation must not produce a result file."""
    target = tmp_path / "result"

    class ExplodingModel:
        def recommend(self, *args, **kwargs):
            raise RuntimeError("model died mid-run")

    with pytest.raises(RuntimeError, match="model died mid-run"):
        evaluate(ExplodingModel(), plan, K=10)
    assert not target.exists()
    assert os.listdir(tmp_path) == []


# --------------------------------------------------- legacy behaviour untouched
def test_legacy_split_default_unchanged(ratings):
    """Without shuffle_random_state the split still uses the global numpy state."""
    np.random.seed(1234)
    before_train, before_test = leave_k_out_split(ratings, K=1, random_state=42)
    np.random.seed(1234)
    after_train, after_test = leave_k_out_split(ratings, K=1, random_state=42)
    assert np.array_equal(before_test.indices, after_test.indices)
    assert np.array_equal(before_train.indices, after_train.indices)

    # and the reproducible path gives a different (but stable) held-out set
    np.random.seed(1234)
    repro_train, repro_test = leave_k_out_split(
        ratings, K=1, random_state=42, shuffle_random_state=42
    )
    np.random.seed(9999)
    repro2_train, repro2_test = leave_k_out_split(
        ratings, K=1, random_state=42, shuffle_random_state=42
    )
    assert np.array_equal(repro_test.indices, repro2_test.indices)
    assert ((repro_train + repro_test) - ratings.tocsr()).nnz == 0


def test_full_catalogue_mode_matches_legacy_metrics(plan):
    """num_negatives=None reproduces ranking_metrics_at_k on the same frozen split."""
    train = plan.train_user_items
    test = plan.test_user_items
    full_plan = EvaluationPlan.from_split(train, test, num_negatives=None, seed=0)
    model = _fitted_model(train)

    result = evaluate(model, full_plan, K=10)
    legacy = ranking_metrics_at_k(model, train, test, K=10, show_progress=False)

    for name in ("precision", "map", "ndcg"):
        assert result.metrics[name] == pytest.approx(legacy[name], rel=0, abs=1e-12)
    assert result.metrics["auc"] == pytest.approx(legacy["auc"], rel=0, abs=1e-12)


def test_default_legacy_split_is_byte_identical_to_recorded_golden():
    """A golden captured before this change must still reproduce exactly.

    Guards "not enabling reproducible mode behaves as before": the default branch
    (no shuffle_random_state) still draws from the global numpy state, in the same
    order, so the historical held-out items are unchanged.
    """
    mat = sparse_random(20, 30, density=0.25, format="csr", random_state=7, dtype=np.float32)
    np.random.seed(123)
    train, test = leave_k_out_split(mat, K=1, random_state=42)

    golden_test_indices = [
        29,
        22,
        21,
        1,
        14,
        12,
        8,
        9,
        19,
        23,
        1,
        14,
        2,
        24,
        17,
        28,
        11,
        4,
        13,
        8,
    ]
    assert test.indices.tolist() == golden_test_indices
    assert test.indptr.tolist() == list(range(21))
    assert (train.nnz, test.nnz) == (130, 20)
    assert ((train + test) - mat).nnz == 0


def test_reproducible_split_is_not_affected_by_global_state():
    """The opt-in path must not read the global numpy state at all."""
    mat = sparse_random(20, 30, density=0.25, format="csr", random_state=7, dtype=np.float32)
    np.random.seed(1)
    a_train, a_test = leave_k_out_split(mat, K=1, random_state=42, shuffle_random_state=42)
    np.random.seed(2)
    b_train, b_test = leave_k_out_split(mat, K=1, random_state=42, shuffle_random_state=42)
    assert np.array_equal(a_test.indices, b_test.indices)
    assert np.array_equal(a_train.indices, b_train.indices)


def test_tie_break_is_by_ascending_item_id(plan):
    """Equal scores must rank by ascending id so ordering never depends on the model."""

    class ConstantScores:
        def recommend(self, userid, user_items, N=10, items=None, **kwargs):
            items = np.asarray(items)
            return items[::-1].copy(), np.zeros(len(items), dtype=np.float32)

    from implicit.reproducible_evaluation import _rank_candidates

    candidates = np.array([5, 3, 9, 1, 7], dtype=np.int32)
    ranked = _rank_candidates(ConstantScores(), 0, None, candidates, K=3)
    assert ranked.tolist() == [1, 3, 5]
