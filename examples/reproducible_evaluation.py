"""End-to-end example for implicit.reproducible_evaluation.

Run this to produce a reproducible evaluation record:

    python examples/reproducible_evaluation.py --out artifacts/repro_eval

It builds one evaluation plan, evaluates two models against the *same* plan, checks
that a second evaluation of the same model is bit-identical, compares the models with a
paired bootstrap, and writes the plan, both results and a comparison record.
"""

import argparse
import json
import os

import numpy as np
from scipy.sparse import random as sparse_random

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ranking_metrics_at_k
from implicit.reproducible_evaluation import (
    build_evaluation_plan,
    compare_results,
    evaluate,
)


def synthetic_ratings(n_users, n_items, nnz, seed):
    density = nnz / float(n_users * n_items)
    ratings = sparse_random(
        n_users, n_items, density=density, format="csr", random_state=seed, dtype=np.float32
    )
    ratings.data = np.ones_like(ratings.data)
    return ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--n-items", type=int, default=5000)
    parser.add_argument("--nnz", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--out", default="artifacts/repro_eval")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ratings = synthetic_ratings(args.n_users, args.n_items, args.nnz, seed=0)

    plan = build_evaluation_plan(
        ratings,
        split="leave_k_out",
        K=1,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )
    plan.save(os.path.join(args.out, "plan.npz"))

    models = {}
    for name, factors in (("als_f32", 32), ("als_f16", 16)):
        model = AlternatingLeastSquares(factors=factors, iterations=10, random_state=42)
        model.fit(plan.train_user_items, show_progress=False)
        models[name] = model

    results = {name: evaluate(model, plan, K=args.K) for name, model in models.items()}
    rerun = evaluate(models["als_f32"], plan, K=args.K, batch_size=1)

    bit_identical = results["als_f32"].metrics == rerun.metrics and all(
        np.array_equal(results["als_f32"].per_user[m], rerun.per_user[m])
        for m in results["als_f32"].per_user
    )

    for name, result in results.items():
        result.save(os.path.join(args.out, name))

    comparison = compare_results(
        results["als_f16"], results["als_f32"], metric="recall", seed=args.seed, num_resamples=2000
    )
    legacy = ranking_metrics_at_k(
        models["als_f32"],
        plan.train_user_items,
        plan.test_user_items,
        K=args.K,
        show_progress=False,
    )

    record = {
        "config": results["als_f32"].config,
        "two_runs_bit_identical": bool(bit_identical),
        "metrics": {name: result.metrics for name, result in results.items()},
        "legacy_full_catalogue_metrics": legacy,
        "comparison_recall": comparison.to_dict(),
        "environment": {
            "numpy": np.__version__,
            "n_users": args.n_users,
            "n_items": args.n_items,
            "nnz": int(ratings.nnz),
        },
    }
    with open(os.path.join(args.out, "record.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)

    print(results["als_f32"].summary())
    print(comparison.summary())
    print(f"two runs bit-identical: {bit_identical}")


if __name__ == "__main__":
    main()
