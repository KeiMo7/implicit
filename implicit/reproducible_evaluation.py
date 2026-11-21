"""Reproducible offline evaluation for implicit-feedback recommender models.

The functions in :mod:`implicit.evaluation` are convenient, but their numbers are
hard to quote with confidence:

* the shuffle in ``leave_k_out_split`` used the global ``numpy`` random state, so a
  seed passed as ``random_state`` did not pin down which interaction was held out;
* metrics were computed against the full catalogue, making the evaluation set
  implicit and, for large catalogues, expensive to reproduce;
* the split policy and the metric definitions were not recorded next to the numbers.

This module adds an opt-in, fully specified evaluation pipeline. It is not used by
the existing evaluation functions or by model training, and importing it does not
change existing behaviour. Everything is enabled by new parameters or new functions;
existing callers keep their current semantics.

Workflow
--------
1. :func:`build_evaluation_plan` (or :meth:`EvaluationPlan.from_split`) builds a
   frozen *evaluation plan*: a train/test split plus, for every evaluated user, a
   fixed list of sampled negative items. The plan records exactly how the split and
   the negatives were produced. Build it once and reuse it for every model you want
   to compare.
2. :func:`evaluate` scores a fitted model against that plan and returns an
   :class:`EvaluationResult` with per-user metrics, aggregate metrics, the complete
   configuration and dataset statistics.
3. :func:`compare_results` reports whether the difference between two models on the
   same plan is larger than sampling noise (paired bootstrap over users), and
   :func:`merge_results` combines sharded runs.

Determinism
-----------
* The split shuffle is seeded with ``shuffle_random_state`` (the split's held-out
  choice) so the split is reproducible from a seed alone.
* Negative sampling uses an independent ``numpy`` stream per user, derived from
  ``SeedSequence([seed, user_id])``. A user's negatives therefore do not depend on
  the order users are processed in, which makes sharding across processes or
  machines safe.
* Users are scored individually in ascending user-id order and aggregated in that
  order. ``batch_size`` only bounds the chunk size (progress reporting / peak
  memory); it does not change any number.
* Bit-identical results assume the same machine, the same BLAS/OpenMP thread
  configuration and the same model object. GPU inference can differ from CPU
  inference in the last floating point bits, and a re-fitted model can differ from a
  previously saved one; comparisons across those boundaries are statistical, not
  bit-identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from .evaluation import leave_k_out_split, train_test_split

__all__ = [
    "EvaluationPlan",
    "EvaluationResult",
    "MetricComparison",
    "build_evaluation_plan",
    "evaluate",
    "compare_results",
    "merge_results",
    "METRIC_DEFINITIONS",
    "SPLIT_LEAVE_K_OUT",
    "SPLIT_RANDOM",
    "SPLIT_EXTERNAL",
]

SPLIT_LEAVE_K_OUT = "leave_k_out"
SPLIT_RANDOM = "random"
SPLIT_EXTERNAL = "external"

#: Human readable definitions of every metric this module reports. They are copied
#: verbatim into :attr:`EvaluationResult.config`, so a result file is self contained
#: and a number can always be traced back to its definition. They intentionally match
#: the definitions used by ``implicit.evaluation.ranking_metrics_at_k``; the only
#: difference is that the mean is macro-averaged over evaluated users here.
METRIC_DEFINITIONS = {
    "precision": (
        "precision@K = hits / min(K, |test_u|), macro-averaged over evaluated users. "
        "The per-user denominator matches implicit.evaluation.ranking_metrics_at_k, "
        "but here the mean is taken over users rather than pooling numerators."
    ),
    "recall": (
        "recall@K = hits / |test_u|, macro-averaged over evaluated users. |test_u| is "
        "the number of held-out items for the user, all of which are candidates."
    ),
    "hit_rate": (
        "hit_rate@K = fraction of evaluated users with at least one held-out item in the top K."
    ),
    "map": (
        "mean average precision@K = mean over users of "
        "sum_{hit at rank r}(hits_up_to_r / r) / min(K, |test_u|)."
    ),
    "ndcg": (
        "normalised discounted cumulative gain@K = DCG@K / IDCG@K with gain 1 per "
        "held-out item, discount 1/log2(rank + 1), and IDCG over min(K, |test_u|) items."
    ),
    "auc": (
        "limited AUC@K: rank based ROC area when only the top K positions are observed "
        "and unobserved positives are treated as tied at the bottom. Negatives are the "
        "sampled candidate negatives (or n_items - |test_u| in full-catalogue mode, "
        "matching implicit.evaluation.ranking_metrics_at_k)."
    ),
}

#: Metric names in a stable order, so result files and shard merges are canonical.
_METRIC_ORDER = tuple(METRIC_DEFINITIONS)


def _seed_for_user(seed: int, user: int) -> np.random.Generator:
    """Return an independent generator for one user.

    The stream depends only on ``seed`` and ``user``, never on processing order, so
    the same plan can be rebuilt on another process, machine or shard and produce
    the same negatives.
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(user)]))


def _sample_user_negatives(user, n_items, excluded, num_negatives, seed):
    """Sample negatives uniformly without replacement, excluding ``excluded``.

    ``excluded`` must contain the user's train **and** test items. Test items are
    excluded explicitly: sampling a held-out item as a negative would put a positive
    in the candidate pool as a distractor and inflate every ranking metric.

    Returns
    -------
    (negatives, available) : (ndarray, int)
        Sorted negative item ids and the number of items that were eligible.
    """
    excluded_arr = np.unique(np.asarray(excluded, dtype=np.int64))
    available = int(n_items) - len(excluded_arr)
    if available <= 0 or num_negatives is None:
        return np.empty(0, dtype=np.int32), max(0, available)

    wanted = min(int(num_negatives), available)
    if wanted <= 0:
        return np.empty(0, dtype=np.int32), available

    rng = _seed_for_user(seed, user)

    # When the excluded set covers most of the catalogue, rejection sampling would
    # thrash. Enumerate the complement once instead; this only triggers for users
    # with a very dense interaction history.
    if available <= 4 * wanted:
        mask = np.ones(int(n_items), dtype=bool)
        mask[excluded_arr] = False
        candidates = np.flatnonzero(mask)
        chosen = rng.choice(candidates, size=wanted, replace=False)
        return np.sort(chosen).astype(np.int32), available

    # Usual case: draw batches from the per-user stream and accept new items until
    # the requested count is reached. Sequential iid draws accepted only when new
    # give a uniform sample without replacement; the final sort makes the result
    # independent of draw order.
    chosen = set()
    while len(chosen) < wanted:
        draw = rng.integers(0, int(n_items), size=max(64, 2 * (wanted - len(chosen))))
        if excluded_arr.size:
            draw = draw[~np.isin(draw, excluded_arr)]
        for item in draw.tolist():
            chosen.add(int(item))
            if len(chosen) == wanted:
                break
    return np.sort(np.fromiter(chosen, dtype=np.int32, count=len(chosen))), available


def _fingerprint(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _counts_stats(values):
    if len(values) == 0:
        return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}
    return {
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": int(np.max(values)),
    }


def _model_description(model):
    """Best-effort JSON description of a model, recorded with the result."""
    description = {"class": type(model).__name__}
    for name in ("factors", "regularization", "alpha", "dtype", "num_threads"):
        if hasattr(model, name):
            value = getattr(model, name)
            if isinstance(value, np.dtype):
                value = value.name
            if isinstance(value, (int, float, str, bool)) or value is None:
                description[name] = value
    return description


def _library_version():
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - version is always present in a real install
        return "unknown"


@dataclass
class EvaluationPlan:
    """A frozen train/test split plus fixed per-user negative samples.

    Build one with :func:`build_evaluation_plan` or :meth:`from_split`. The plan is
    the unit that stays constant while models change; its :attr:`fingerprint` changes
    if the split, the seed, the negative count or the sampled negatives change, so
    two evaluations can be proven comparable before their numbers are compared.

    Attributes
    ----------
    train_user_items : csr_matrix
        Interactions used to fit the model and to score users.
    test_user_items : csr_matrix
        Held-out interactions. Only users with at least one entry are evaluated.
    negative_indptr, negative_indices : ndarray
        CSR-like structure holding the sampled negative item ids for each entry of
        :attr:`user_ids`. ``None`` when ``num_negatives`` is ``None`` (full-catalogue
        mode).
    insufficient_negative_users : ndarray
        Users that received fewer than ``num_negatives`` negatives because their
        train plus test items nearly cover the catalogue.
    """

    train_user_items: csr_matrix
    test_user_items: csr_matrix
    seed: int = 0
    num_negatives: Optional[int] = 100
    split: str = SPLIT_EXTERNAL
    split_params: dict = field(default_factory=dict)
    negative_indptr: Optional[np.ndarray] = None
    negative_indices: Optional[np.ndarray] = None
    insufficient_negative_users: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    user_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    fingerprint: str = ""

    def __post_init__(self):
        self._user_position = {int(u): i for i, u in enumerate(self.user_ids)}

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_split(
        cls,
        train_user_items,
        test_user_items,
        num_negatives=100,
        seed=0,
        split=SPLIT_EXTERNAL,
        split_params=None,
    ):
        """Build a plan from an already prepared train/test split.

        Use this for protocols this module does not implement, for example a time
        based split where the last interaction of each user is held out, or a split
        shared by an earlier experiment. Negative sampling and the recorded
        configuration are identical to the built-in splits: negatives are drawn
        uniformly without replacement from items that are in neither the user's train
        row nor the user's test row, so held-out items can never become negatives.
        """
        train = csr_matrix(train_user_items).tocsr()
        test = csr_matrix(test_user_items).tocsr()
        if train.shape != test.shape:
            raise ValueError("train_user_items and test_user_items must have the same shape")
        if num_negatives is not None and int(num_negatives) < 0:
            raise ValueError("num_negatives must be >= 0 or None")
        train.sort_indices()
        test.sort_indices()

        n_items = int(train.shape[1])
        user_ids = np.flatnonzero(np.diff(test.indptr) > 0).astype(np.int32)

        negative_indptr = None
        negative_indices = None
        insufficient = []
        if num_negatives is not None:
            num_negatives = int(num_negatives)
            per_user = []
            counts = np.zeros(len(user_ids) + 1, dtype=np.int64)
            for idx, user in enumerate(user_ids):
                user = int(user)
                train_row = train.indices[train.indptr[user] : train.indptr[user + 1]]
                test_row = test.indices[test.indptr[user] : test.indptr[user + 1]]
                excluded = np.concatenate((train_row, test_row))
                negatives, _ = _sample_user_negatives(user, n_items, excluded, num_negatives, seed)
                if len(negatives) < num_negatives:
                    insufficient.append(user)
                per_user.append(negatives)
                counts[idx + 1] = len(negatives)
            negative_indptr = counts.cumsum().astype(np.int64)
            negative_indices = (
                np.concatenate(per_user).astype(np.int32)
                if per_user
                else np.empty(0, dtype=np.int32)
            )
            if insufficient:
                warnings.warn(
                    f"{len(insufficient)} user(s) had fewer than {num_negatives} eligible "
                    "negatives because their train and test items already cover nearly the "
                    "whole catalogue. Those users are still evaluated with the negatives "
                    "that were available. A smaller negative pool makes it easier for a "
                    "held-out item to land in the top K, so recall/precision/hit_rate/NDCG/"
                    "AUC are biased UPWARDS for them; the aggregate metrics are biased "
                    "upwards in proportion to how many users are affected.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        plan = cls(
            train_user_items=train,
            test_user_items=test,
            seed=int(seed),
            num_negatives=None if num_negatives is None else int(num_negatives),
            split=split,
            split_params=dict(split_params or {}),
            negative_indptr=negative_indptr,
            negative_indices=negative_indices,
            insufficient_negative_users=np.asarray(insufficient, dtype=np.int32),
            user_ids=user_ids,
        )
        plan._finalize()
        return plan

    # ---------------------------------------------------------------- validation
    def _finalize(self):
        if self.num_negatives is None or self.negative_indices is None:
            negatives = b""
        else:
            negatives = self.negative_indices.tobytes() + self.negative_indptr.tobytes()
        payload = b"|".join(
            [
                self.split.encode(),
                json.dumps(self.split_params, sort_keys=True, default=str).encode(),
                str(self.seed).encode(),
                str(self.num_negatives).encode(),
                self.train_user_items.indptr.tobytes(),
                self.train_user_items.indices.tobytes(),
                self.train_user_items.data.tobytes(),
                self.test_user_items.indptr.tobytes(),
                self.test_user_items.indices.tobytes(),
                self.test_user_items.data.tobytes(),
                negatives,
            ]
        )
        self.fingerprint = _fingerprint(payload)
        self._user_position = {int(u): i for i, u in enumerate(self.user_ids)}

    # ------------------------------------------------------------------ summary
    def summary(self):
        """Return the scale and distribution of the evaluation set.

        The dictionary is JSON serialisable and is embedded in every
        :class:`EvaluationResult`, so a reported number can always be checked against
        the dataset it came from (users, items, interactions, density, per-user test
        and negative counts, skipped users).
        """
        train = self.train_user_items
        test = self.test_user_items
        n_users, n_items = train.shape
        per_user_test = np.diff(test.indptr)
        evaluated = per_user_test[self.user_ids] if len(self.user_ids) else np.empty(0, dtype=int)
        total_possible = float(n_users) * float(n_items)

        if self.num_negatives is not None and self.negative_indptr is not None:
            negative_counts = np.diff(self.negative_indptr)
            no_neg = int(np.count_nonzero(negative_counts == 0))
        else:
            negative_counts = np.empty(0, dtype=np.int64)
            no_neg = 0

        positive_total = int(np.sum(evaluated))
        return {
            "n_users": int(n_users),
            "n_items": int(n_items),
            "n_interactions": int(train.nnz + test.nnz),
            "train_nnz": int(train.nnz),
            "test_nnz": int(test.nnz),
            "density": float((train.nnz + test.nnz) / total_possible) if total_possible else 0.0,
            "sparsity": (
                1.0 - float((train.nnz + test.nnz) / total_possible) if total_possible else 0.0
            ),
            "n_items_in_test": int(np.count_nonzero(per_user_test)),
            "test_items_per_user": _counts_stats(evaluated),
            "evaluated_users": int(len(self.user_ids)),
            "positive_nnz": positive_total,
            "negatives_per_user": (
                _counts_stats(negative_counts) if self.num_negatives is not None else None
            ),
            "insufficient_negatives_users": int(len(self.insufficient_negative_users)),
            "skipped_users": {
                "no_test_items": int(n_users - len(self.user_ids)),
                "no_negatives": no_neg,
            },
        }

    def config(self):
        """Return the complete, JSON serialisable evaluation configuration."""
        sampled = self.num_negatives is not None
        return {
            "mode": "reproducible_sampled" if sampled else "reproducible_full_catalogue",
            "split": self.split,
            "split_params": dict(self.split_params),
            "seed": int(self.seed),
            "num_negatives": self.num_negatives,
            "sampling": "uniform" if sampled else None,
            "negative_sampling": (
                "uniform without replacement over items not in the user's train or test "
                "items; per-user stream SeedSequence([seed, user_id])"
                if sampled
                else "none (rank against the full catalogue)"
            ),
            "negatives_exclude": (["train items", "test items"] if sampled else []),
            "test_items_are_negatives": False,
            "metric_definitions": METRIC_DEFINITIONS,
            "plan_fingerprint": self.fingerprint,
            "dataset": self.summary(),
        }

    def negatives_for(self, user):
        """Return the sampled negative item ids for ``user`` (empty in full mode)."""
        if self.num_negatives is None or int(user) not in self._user_position:
            return np.empty(0, dtype=np.int32)
        pos = self._user_position[int(user)]
        return self.negative_indices[self.negative_indptr[pos] : self.negative_indptr[pos + 1]]

    # -------------------------------------------------------------- persistence
    def save(self, path):
        """Save the plan atomically to ``path`` (a single ``.npz`` file).

        The write goes to a temporary file in the same directory and is renamed into
        place only once complete, so an interrupted save never leaves a half-written
        plan behind and never destroys an existing plan.
        """
        path = os.fspath(path)
        if not path.endswith(".npz"):
            path += ".npz"
        tmp = f"{path}.tmp-{os.getpid()}"
        train, test = self.train_user_items, self.test_user_items
        try:
            np.savez(
                tmp,
                meta=np.array(json.dumps(self.config())),
                train_indptr=train.indptr,
                train_indices=train.indices,
                train_data=train.data,
                test_indptr=test.indptr,
                test_indices=test.indices,
                test_data=test.data,
                shape=np.array(train.shape),
                user_ids=self.user_ids,
                negative_indptr=(
                    self.negative_indptr
                    if self.negative_indptr is not None
                    else np.empty(0, dtype=np.int64)
                ),
                negative_indices=(
                    self.negative_indices
                    if self.negative_indices is not None
                    else np.empty(0, dtype=np.int32)
                ),
                insufficient_negative_users=self.insufficient_negative_users,
            )
            os.replace(tmp + ".npz", path)
        except BaseException:
            for candidate in (tmp, tmp + ".npz"):
                if os.path.exists(candidate):
                    os.unlink(candidate)
            raise

    @classmethod
    def load(cls, path):
        """Load a plan saved with :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            shape = tuple(int(x) for x in data["shape"])
            train = csr_matrix(
                (data["train_data"], data["train_indices"], data["train_indptr"]), shape=shape
            )
            test = csr_matrix(
                (data["test_data"], data["test_indices"], data["test_indptr"]), shape=shape
            )
            full = meta["num_negatives"] is not None
            plan = cls(
                train_user_items=train,
                test_user_items=test,
                seed=meta["seed"],
                num_negatives=meta["num_negatives"],
                split=meta["split"],
                split_params=meta["split_params"],
                negative_indptr=(data["negative_indptr"] if full else None),
                negative_indices=(data["negative_indices"] if full else None),
                insufficient_negative_users=data["insufficient_negative_users"],
                user_ids=data["user_ids"],
            )
        plan._finalize()
        if plan.fingerprint != meta["plan_fingerprint"]:
            raise ValueError("Loaded plan does not match its recorded fingerprint")
        return plan


def build_evaluation_plan(
    ratings,
    split=SPLIT_LEAVE_K_OUT,
    K=1,
    train_percentage=0.8,
    num_negatives=100,
    seed=0,
):
    """Split ``ratings`` and sample negatives, returning an :class:`EvaluationPlan`.

    Parameters
    ----------
    ratings : coo_matrix or csr_matrix
        User by item interaction matrix.
    split : {"leave_k_out", "random"}
        ``"leave_k_out"`` holds out the last ``K`` interactions of every eligible user
        (the ``implicit.evaluation.leave_k_out_split`` protocol). ``"random"`` uses
        ``implicit.evaluation.train_test_split`` with ``train_percentage``. An
        externally produced split must go through :meth:`EvaluationPlan.from_split`.
    K : int
        Number of interactions held out per user for the leave-k-out split.
    train_percentage : float
        Training fraction for the random split.
    num_negatives : int or None
        Negatives sampled per evaluated user. ``None`` ranks against the full
        catalogue (the historical behaviour).
    seed : int
        Seed controlling both the held-out choice of the shuffle split and the
        negative sampling. Passing the same seed reproduces the same plan.

    Notes
    -----
    ``seed`` is used for two independent purposes: ``leave_k_out_split`` receives it
    as ``random_state`` (user eligibility / train-only selection) and as
    ``shuffle_random_state`` (which interaction is held out), and negative sampling
    derives one stream per user from it. The two RNG consumers are separate streams,
    so changing one policy does not silently move the other.
    """
    ratings = csr_matrix(ratings).tocsr()
    ratings.sort_indices()
    if split == SPLIT_LEAVE_K_OUT:
        # shuffle_random_state pins the held-out selection; without it the shuffle
        # would still fall back to the global numpy state (historical behaviour).
        train, test = leave_k_out_split(ratings, K=K, random_state=seed, shuffle_random_state=seed)
        split_params = {"K": int(K)}
    elif split == SPLIT_RANDOM:
        train, test = train_test_split(
            ratings, train_percentage=train_percentage, random_state=seed
        )
        split_params = {"train_percentage": float(train_percentage)}
    elif split == SPLIT_EXTERNAL:
        raise ValueError("Use EvaluationPlan.from_split for an externally produced split")
    else:
        raise ValueError(f"Unknown split {split!r}")

    return EvaluationPlan.from_split(
        train,
        test,
        num_negatives=num_negatives,
        seed=seed,
        split=split,
        split_params=split_params,
    )


def _user_metrics(ranked_ids, positive_ids, K, num_neg_items):
    """Compute one user's metrics from a score-ordered candidate list.

    ``positive_ids`` is a set of the user's held-out items. ``num_neg_items`` is the
    number of distractors in the candidate pool: the sampled negatives, or
    ``n_items - |test_u|`` in full-catalogue mode.
    """
    n_pos = len(positive_ids)
    top = ranked_ids[:K]

    hits = 0
    ap = 0.0
    dcg = 0.0
    miss = 0
    auc = 0.0
    for rank in range(len(top)):
        if int(top[rank]) in positive_ids:
            hits += 1
            ap += hits / (rank + 1)
            dcg += 1.0 / np.log2(rank + 2)
        else:
            miss += 1
            auc += hits

    denom = min(K, n_pos) if n_pos else 1
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, min(K, n_pos) + 2)))) if n_pos else 0.0
    if n_pos and num_neg_items > 0:
        # Same limited-AUC formula as implicit.evaluation.ranking_metrics_at_k:
        # observed positives contribute 1 per preceding negative, unobserved positives
        # are tied at the bottom.
        auc += ((hits + n_pos) / 2.0) * (num_neg_items - miss)
        auc /= n_pos * num_neg_items
    else:
        auc = 0.0

    return {
        "precision": hits / denom,
        "recall": hits / n_pos if n_pos else 0.0,
        "hit_rate": 1.0 if hits else 0.0,
        "map": ap / denom if n_pos else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "auc": auc,
    }


def _rank_candidates(model, user, train_row, candidates, K):
    """Return the candidate ids ordered by descending score with a stable tie break.

    Ranking is restricted to the plan's candidate set through the model's public
    ``recommend(items=...)`` interface, so the same function works for ALS, BPR, LMF
    and item-item models. Ties are broken by ascending item id, so the ranking does
    not depend on the order a model happens to return tied items in.
    """
    ids, scores = model.recommend(
        user,
        train_row,
        N=len(candidates),
        items=candidates,
        filter_already_liked_items=True,
    )
    ids = np.asarray(ids).ravel()
    scores = np.asarray(scores).ravel()
    order = np.lexsort((ids, -scores))
    return ids[order][:K]


def evaluate(model, plan: EvaluationPlan, K=10, show_progress=False, batch_size=1000, users=None):
    """Evaluate a fitted ``model`` against a fixed :class:`EvaluationPlan`.

    Users are scored individually so that each user's ranking depends only on the
    model and that user's candidate set. ``batch_size`` groups users for progress
    reporting and bounds how many per-user metric values are appended between
    reductions; it does not change any metric. That is what makes a batched run
    produce the same numbers as an unbatched one.

    ``users`` optionally restricts evaluation to a subset of user ids in the plan. It
    exists so disjoint shards can be evaluated against the *same* plan (same
    fingerprint) and then combined with :func:`merge_results`; the merged result is
    identical to one run over the union. Users requested here that have no held-out
    item are ignored, and the plan-level "no test items" count in the config stays
    the full-plan count.

    A user with no sampled negative is skipped and counted in
    ``config["skipped_users"]["no_negatives"]``; skipped users are never averaged in
    as zeros.

    Returns
    -------
    EvaluationResult
    """
    if not isinstance(plan, EvaluationPlan):
        raise TypeError("plan must be an EvaluationPlan built by build_evaluation_plan")
    K = int(K)
    if K < 1:
        raise ValueError("K must be >= 1")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    train = plan.train_user_items
    test = plan.test_user_items
    n_items = int(train.shape[1])

    all_users = np.asarray(plan.user_ids, dtype=np.int32)
    # Plan-level count: users with no held-out item. Independent of the `users`
    # filter, so merging shards never multiplies it.
    skipped_no_test = int(train.shape[0] - len(all_users))
    is_full_run = users is None
    requested = None
    if is_full_run:
        users = all_users
    else:
        requested = np.asarray(users)
        users = np.intersect1d(all_users, requested.astype(all_users.dtype)).astype(np.int32)

    per_user = {name: [] for name in _METRIC_ORDER}
    evaluated = []
    skipped_no_negatives = 0

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm as _tqdm

            progress = _tqdm(total=len(users))
        except Exception:  # pragma: no cover - tqdm is a dependency, but degrade safely
            progress = None

    for start in range(0, len(users), batch_size):
        chunk = users[start : start + batch_size]
        for user in chunk:
            user = int(user)
            positives = test.indices[test.indptr[user] : test.indptr[user + 1]]
            positive_ids = set(int(x) for x in positives)
            if not positive_ids:
                # Cannot happen for users taken from the plan, but keep the guard
                # for plans assembled by hand.
                continue

            train_row = train[user]
            if plan.num_negatives is None:
                ids, _ = model.recommend(user, train_row, N=K, filter_already_liked_items=True)
                ranked = np.asarray(ids).ravel()[:K]
                num_neg_items = n_items - len(positive_ids)
            else:
                negatives = plan.negatives_for(user)
                if len(negatives) == 0:
                    skipped_no_negatives += 1
                    continue
                candidates = np.union1d(positives, negatives).astype(np.int32)
                ranked = _rank_candidates(model, user, train_row, candidates, K)
                num_neg_items = int(len(negatives))

            metrics = _user_metrics(ranked, positive_ids, K, num_neg_items)
            for name in _METRIC_ORDER:
                per_user[name].append(metrics[name])
            evaluated.append(user)
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    if not evaluated:
        raise ValueError(
            "No user has an evaluable sample; check the split or the negative sampling policy"
        )

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in per_user.items()}
    metrics = {name: float(np.sum(values) / len(values)) for name, values in arrays.items()}

    config = plan.config()
    config.update(
        {
            "K": int(K),
            "model": _model_description(model),
            "batch_size": int(batch_size),
            "library_version": _library_version(),
            "evaluated_users": int(len(evaluated)),
            "skipped_users": {
                "no_test_items": int(skipped_no_test),
                "no_negatives": int(skipped_no_negatives),
            },
            "user_shard": (
                None
                if is_full_run
                else {"n_requested": int(len(requested)), "n_evaluated": int(len(evaluated))}
            ),
            "aggregation": "mean over evaluated users, users summed in ascending id order",
        }
    )

    return EvaluationResult(
        metrics=metrics,
        per_user=arrays,
        user_ids=np.asarray(evaluated, dtype=np.int32),
        config=config,
        plan=plan,
    )


@dataclass
class EvaluationResult:
    """Metrics plus the configuration and per-user values that produced them."""

    metrics: dict
    per_user: dict
    user_ids: np.ndarray
    config: dict
    plan: Optional[EvaluationPlan] = None

    def to_dict(self):
        """Return a JSON serialisable view of the result (without per-user arrays)."""
        return {"metrics": dict(self.metrics), "config": dict(self.config)}

    def summary(self):
        """Human readable multi-line summary of metrics and configuration."""
        lines = ["metrics:"]
        for name in _METRIC_ORDER:
            lines.append(f"  {name:10s} {self.metrics[name]:.10f}")
        cfg = self.config
        lines.append(
            f"split={cfg['split']} K={cfg['K']} seed={cfg['seed']} "
            f"num_negatives={cfg['num_negatives']}"
        )
        ds = cfg["dataset"]
        lines.append(
            f"evaluated_users={cfg['evaluated_users']} "
            f"n_users={ds['n_users']} n_items={ds['n_items']} density={ds['density']:.6f}"
        )
        lines.append(
            f"skipped no_test_items={cfg['skipped_users']['no_test_items']} "
            f"no_negatives={cfg['skipped_users']['no_negatives']}"
        )
        lines.append(f"plan_fingerprint={cfg['plan_fingerprint']}")
        return "\n".join(lines)

    def save(self, path):
        """Write the result atomically as an ``.npz`` and a sidecar ``.json`` file.

        Both files are written through temporary files and renamed into place only
        after they are complete, so a failed or interrupted evaluation/save never
        leaves a truncated result and never overwrites an existing good result with a
        partial one. Call this only after ``evaluate`` returned; if evaluation itself
        raises, no file is touched.
        """
        path = os.fspath(path)
        stem = path[:-4] if path.endswith(".npz") else path
        npz_path = stem + ".npz"
        json_path = stem + ".json"
        tmp_npz = f"{npz_path}.tmp-{os.getpid()}"
        tmp_json = f"{json_path}.tmp-{os.getpid()}"
        try:
            np.savez(
                tmp_npz,
                meta=np.array(json.dumps(self.to_dict())),
                user_ids=self.user_ids,
                **{f"metric_{k}": v for k, v in self.per_user.items()},
            )
            with open(tmp_json, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_npz + ".npz", npz_path)
            os.replace(tmp_json, json_path)
        except BaseException:
            for candidate in (tmp_npz, tmp_npz + ".npz", tmp_json):
                if os.path.exists(candidate):
                    os.unlink(candidate)
            raise

    @classmethod
    def load(cls, path):
        """Load a result saved with :meth:`save`."""
        path = os.fspath(path)
        stem = path[:-4] if path.endswith(".npz") else path
        with open(stem + ".json", encoding="utf-8") as handle:
            meta = json.load(handle)
        with np.load(stem + ".npz", allow_pickle=False) as data:
            user_ids = data["user_ids"]
            per_user = {
                name: data[f"metric_{name}"] for name in _METRIC_ORDER if f"metric_{name}" in data
            }
        return cls(
            metrics=dict(meta["metrics"]),
            per_user=per_user,
            user_ids=user_ids,
            config=dict(meta["config"]),
        )


@dataclass
class MetricComparison:
    """Result of comparing two evaluations that share the same plan."""

    metric: str
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_users: int
    significant: bool
    method: str

    def to_dict(self):
        """Return a JSON serialisable view of the comparison."""
        return {
            "metric": self.metric,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n_users": self.n_users,
            "significant": self.significant,
            "method": self.method,
        }

    def summary(self):
        flag = "significant" if self.significant else "not significant"
        return (
            f"{self.metric}: delta={self.delta:+.6f} "
            f"95% CI [{self.ci_low:+.6f}, {self.ci_high:+.6f}] "
            f"p={self.p_value:.4f} n={self.n_users} ({flag})"
        )


def compare_results(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    metric="recall",
    seed=0,
    num_resamples=10000,
    alpha=0.05,
):
    """Paired bootstrap comparison of two evaluations on the same plan.

    Both results must come from the same plan, otherwise the candidate sets differ
    and the comparison is meaningless. Per-user differences are bootstrapped over
    users, which respects the paired design: the same user, the same candidates,
    two models. The p-value is the two-sided bootstrap tail probability, and the
    interval is the percentile interval; ``significant`` additionally requires the
    interval to exclude zero.
    """
    if metric not in baseline.per_user or metric not in candidate.per_user:
        raise ValueError(f"Unknown metric {metric!r}")
    if baseline.config["plan_fingerprint"] != candidate.config["plan_fingerprint"]:
        raise ValueError(
            "The two results were produced with different evaluation plans; "
            "rebuild both with the same plan before comparing"
        )
    if baseline.config.get("K") != candidate.config.get("K"):
        raise ValueError("The two results used different K values")
    if not np.array_equal(baseline.user_ids, candidate.user_ids):
        raise ValueError("The two results cover different users")
    num_resamples = int(num_resamples)
    if num_resamples < 1:
        raise ValueError("num_resamples must be >= 1")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    diffs = np.asarray(candidate.per_user[metric], dtype=np.float64) - np.asarray(
        baseline.per_user[metric], dtype=np.float64
    )
    n = len(diffs)
    if n == 0:
        raise ValueError("The two results have no evaluated users to compare")
    observed = float(np.mean(diffs))

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xB007]))
    means = np.empty(num_resamples, dtype=np.float64)
    for i in range(num_resamples):
        means[i] = diffs[rng.integers(0, n, size=n)].mean()

    ci_low, ci_high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    below = np.count_nonzero(means <= 0) / num_resamples
    above = np.count_nonzero(means >= 0) / num_resamples
    p_value = min(1.0, 2.0 * min(below, above))

    return MetricComparison(
        metric=metric,
        delta=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        n_users=n,
        significant=bool(p_value < alpha and not (ci_low <= 0 <= ci_high)),
        method=f"paired bootstrap over users, {num_resamples} resamples, seed={seed}",
    )


def merge_results(results):
    """Merge results computed on disjoint user shards of the same plan.

    Each shard must be produced with :func:`evaluate` on the same plan and the same
    ``K``. Users are sorted into canonical order before the mean is taken, so merging
    is deterministic regardless of the order shards are supplied in, and the merged
    result equals a single run over the union of the shards.

    Because negative sampling uses a per-user stream, a plan rebuilt on each machine
    has the same negatives, so sharded results are comparable.
    """
    results = list(results)
    if not results:
        raise ValueError("merge_results needs at least one result")
    if len({r.config["plan_fingerprint"] for r in results}) != 1:
        raise ValueError("All results must share the same plan fingerprint")
    if len({r.config.get("K") for r in results}) != 1:
        raise ValueError("All results must use the same K")
    if len({bool(r.config.get("user_shard")) for r in results}) != 1:
        raise ValueError("All results must either be shards or all full runs")

    users = np.concatenate([r.user_ids for r in results])
    if len(np.unique(users)) != len(users):
        raise ValueError("Shards overlap; each user must appear in exactly one shard")
    order = np.argsort(users, kind="stable")
    users = users[order]

    per_user = {}
    for name in _METRIC_ORDER:
        per_user[name] = np.concatenate([r.per_user[name] for r in results])[order]
    metrics = {name: float(np.sum(values) / len(values)) for name, values in per_user.items()}

    config = dict(results[0].config)
    config["evaluated_users"] = int(len(users))
    # no_test_items is a plan-level property: every shard reports the same value, so
    # take it from the first result rather than summing. no_negatives is per user and
    # shards are disjoint, so sum it.
    config["skipped_users"] = {
        "no_test_items": int(results[0].config["skipped_users"]["no_test_items"]),
        "no_negatives": int(sum(r.config["skipped_users"]["no_negatives"] for r in results)),
    }
    config["user_shard"] = None
    config["merged_shards"] = len(results)

    return EvaluationResult(
        metrics=metrics,
        per_user=per_user,
        user_ids=users.astype(np.int32),
        config=config,
        plan=results[0].plan,
    )
