Reproducible Evaluation
=======================

.. automodule:: implicit.reproducible_evaluation

Why the numbers moved between runs
----------------------------------

Two things in the historical evaluation path were not pinned down by a seed:

1. ``leave_k_out_split(..., random_state=seed)`` seeded the user-eligibility
   draws, but the step that chooses *which* interaction of a user is held out
   (``_take_tails(..., shuffled=True)``) drew from the **global** ``numpy``
   random state. Two runs with the same ``random_state`` therefore produced
   different splits unless the caller also called ``np.random.seed`` first.
2. Metrics in :func:`implicit.evaluation.ranking_metrics_at_k` rank against the
   full catalogue. There was no explicit, samplable negative set, so the
   evaluated candidate pool differed between runs (and between catalogue
   versions) without being recorded.

``implicit.reproducible_evaluation`` fixes both by making the split and the
negative sample part of a saved *plan*, and records the plan's fingerprint and
its construction rules next to every metric.

Quick start
-----------

.. code-block:: python

    import implicit
    from implicit.reproducible_evaluation import (
        build_evaluation_plan,
        compare_results,
        evaluate,
    )

    # 1. Freeze the evaluation set once. Reuse the same plan for every model.
    plan = build_evaluation_plan(
        ratings,                 # user x item csr/coo matrix
        split="leave_k_out",     # or "random"
        K=1,                     # leave-one-out
        num_negatives=100,       # per-user sampled distractors
        seed=20240101,           # controls split shuffle *and* sampling
    )
    plan.save("eval_plan.npz")   # atomic; load with EvaluationPlan.load()

    # 2. Evaluate any fitted model against the frozen plan.
    result_a = evaluate(model_a, plan, K=10, batch_size=1000)
    result_b = evaluate(model_b, plan, K=10, batch_size=1000)

    print(result_a.summary())
    result_a.save("result_a")    # writes result_a.npz + result_a.json atomically

    # 3. Is the difference real?
    print(compare_results(result_a, result_b, metric="recall").summary())

The same plan can be rebuilt from the seed on another machine (or a shard of
one) and produces the same split and the same negatives; results from disjoint
shards are combined with :func:`merge_results` and equal a single run.

Parameters and defaults
-----------------------

``build_evaluation_plan(ratings, split="leave_k_out", K=1,
train_percentage=0.8, num_negatives=100, seed=0)``

``split``
    ``"leave_k_out"`` (hold out the last ``K`` interactions of each eligible
    user, the existing :func:`implicit.evaluation.leave_k_out_split` protocol)
    or ``"random"`` (:func:`implicit.evaluation.train_test_split` with
    ``train_percentage``). A split produced elsewhere (for example a time
    split) goes through ``EvaluationPlan.from_split``.
``num_negatives``
    Negatives per evaluated user; ``None`` ranks against the full catalogue.
    The value is recorded in the result and **must match** when comparing
    numbers across models or versions; sampled and full-catalogue numbers are
    not comparable.
``seed``
    Controls the split shuffle and the per-user negative stream. The same seed
    reproduces the same plan and therefore the same metrics.

``evaluate(model, plan, K=10, show_progress=False, batch_size=1000, users=None)``

``batch_size``
    Chunk size for progress reporting and memory bounding; **does not change
    any metric**. ``users`` restricts evaluation to a subset of plan users so
    that disjoint shards can run in parallel; use :func:`merge_results` to
    combine them.

Split and sampling rules, recorded in every result
--------------------------------------------------

* **Split.** ``leave_k_out`` holds out ``K`` interactions per eligible user;
  the interaction is chosen by ``shuffle_random_state=seed`` inside
  :func:`implicit.evaluation.leave_k_out_split`. ``random`` holds out each
  interaction with probability ``1 - train_percentage``.
* **Negatives.** Uniform without replacement from items that are in neither the
  user's train row nor the user's test row. Test items are excluded
  **explicitly**: sampling a held-out item as a distractor would inflate every
  ranking metric. Each user has an independent stream,
  ``SeedSequence([seed, user_id])``, so sampling does not depend on user order
  or sharding.
* **Metrics.** ``recall``, ``hit_rate``, ``ndcg``, ``precision``, ``map`` and
  ``auc`` are defined in :data:`METRIC_DEFINITIONS`; the definitions are copied
  into ``EvaluationResult.config["metric_definitions"]`` and into the saved
  JSON. ``precision``/``map`` use the same per-user denominators as
  ``ranking_metrics_at_k``; the aggregate is a macro average over users that
  have a held-out item.
* **Scale and distribution.** ``config["dataset"]`` records ``n_users``,
  ``n_items``, ``n_interactions``, ``density``, per-user test counts,
  per-user negative counts and the number of skipped users, so a reported
  metric can be checked against the data it came from.

Failure and degradation
-----------------------

* A user with **no held-out item** is not evaluated; the count is reported in
  ``config["skipped_users"]["no_test_items"]``.
* A user with **no eligible negative** (train plus test items cover the whole
  catalogue) is skipped and counted in
  ``config["skipped_users"]["no_negatives"]``. Skipped users are never averaged
  in as zeros.
* If a user gets **fewer negatives than requested**, a ``RuntimeWarning`` names
  the count and states the direction of the bias: a smaller distractor pool
  makes it easier for a held-out item to land in the top ``K``, so metrics are
  biased **upwards** for those users and, in proportion, in the aggregate.
* If no user is evaluable, ``evaluate`` raises ``ValueError`` instead of
  returning a meaningless number.
* A failed or interrupted ``save`` (plan or result) writes through temporary
  files and removes them on error, so no truncated result file is left behind.
  If ``evaluate`` itself raises, no file is touched.

Determinism, batching and multiple machines
-------------------------------------------

* Same model object + same plan + same BLAS/OpenMP configuration → identical
  per-user and aggregate numbers, regardless of ``batch_size`` or sharding.
  There are tests asserting byte equality of the per-user arrays.
* **No model training code and no model weights change.** This module only
  reads a fitted model through its public ``recommend(items=...)`` interface,
  so it works with ALS, BPR, LMF and item-item models, CPU or GPU.
* Batching adds no data copies: each user ranks only ``|test_u| +
  num_negatives`` candidates instead of the whole catalogue, so peak memory
  does not grow with ``num_negatives``.
* For multiple machines, ship the plan file (or rebuild it from the same seed)
  and evaluate disjoint ``users`` slices. No cross-shard synchronisation is
  needed; :func:`merge_results` sorts users into canonical order, so the result
  does not depend on shard order. If a shard fails, merge the shards that
  succeeded and record the missing users in ``config["merged_shards"]`` and the
  user list — do not substitute an empty shard, which would silently change
  the mean.
* Bit-identical guarantees hold for the same machine and the same
  BLAS/OpenMP thread configuration. Different thread counts can change the last
  floating-point bits of the model's scores; set ``OPENBLAS_NUM_THREADS=1``
  (the library already recommends this) for stable runs. GPU and CPU inference
  can differ in the last bits; comparisons across that boundary are
  statistical, not bit-identical.

Compatibility boundaries
------------------------

* Existing callers are unaffected. ``leave_k_out_split`` gained
  ``shuffle_random_state=None``; ``None`` keeps the historical global-RNG
  shuffle, and ``random_state`` keeps its existing meaning.
* Existing models/weights are unaffected: this is evaluation only.
* Results produced with the old full-catalogue evaluator are comparable only to
  ``evaluate(plan_with_num_negatives=None)`` on the same split; they are not
  comparable to sampled results. Old *splits* were not reproducible from a
  seed, so re-run the split through a plan before quoting a number.
* No environment variables or global state; all configuration is function
  arguments with defaults. The module is never invoked by existing code paths.

API
---

.. autoclass:: implicit.reproducible_evaluation.EvaluationPlan
   :members:

.. autoclass:: implicit.reproducible_evaluation.EvaluationResult
   :members:

.. autoclass:: implicit.reproducible_evaluation.MetricComparison
   :members:

.. autofunction:: implicit.reproducible_evaluation.build_evaluation_plan

.. autofunction:: implicit.reproducible_evaluation.evaluate

.. autofunction:: implicit.reproducible_evaluation.compare_results

.. autofunction:: implicit.reproducible_evaluation.merge_results

.. autodata:: implicit.reproducible_evaluation.METRIC_DEFINITIONS
