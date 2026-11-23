"""独立重排包装层。

:class:`RerankingRecommender` 包装任意具备 ``recommend(userid, user_items, ...)``
接口的召回/打分模型（鸭子类型，ALS/BPR/LMF/KNN/ANN 均可），在其输出与调用方
之间插入重排层。被包装模型的调用方式与返回结构完全不变：

* ``recommend`` 签名与 :class:`implicit.recommender_base.RecommenderBase` 一致，
  单用户返回一维 ``(ids, scores)``，批量用户返回二维 ``(ids, scores)``；
* 模型推理与打分路径完全不动，分数原样透传（dtype 也保持一致）；
* ``enabled=False`` 时直接返回底层模型的原始数组，逐条一致；
* 重排引擎可独立替换（构造时注入，或替换 ``engine`` 属性接入别的召回来源）。
"""

import numpy as np

from .engine import RerankEngine


class RerankingRecommender:
    """在召回模型外包裹一层展示重排。

    Parameters
    ----------
    model : object
        被包装的推荐模型，只需实现与 ``RecommenderBase`` 兼容的 ``recommend``。
    engine : RerankEngine
        重排引擎。
    enabled : bool, 可选
        重排总开关，默认 True。False 时完全透传，输出与底层模型逐条一致。
    recall_oversample : int, 可选
        回填用的召回扩大倍数：底层按 ``N * recall_oversample`` 召回，前 N 个
        作为主候选，其余作为可追溯的回填池。默认 2，设为 1 表示不扩大。
    """

    def __init__(self, model, engine, enabled=True, recall_oversample=2):
        if not hasattr(model, "recommend"):
            raise TypeError("被包装对象必须实现 recommend(userid, user_items, ...)")
        if not isinstance(engine, RerankEngine):
            raise TypeError("engine 必须是 RerankEngine 实例（可独立替换）")
        if recall_oversample < 1:
            raise ValueError("recall_oversample 必须 >= 1")
        self.model = model
        self.engine = engine
        self.enabled = enabled
        self.recall_oversample = recall_oversample

        # 最近一次请求的逐条报告，与返回列表同源；批量时按用户下标索引
        self.last_report = None

    # ------------------------------------------------------------------
    def recommend(
        self,
        userid,
        user_items,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
        items=None,
    ):
        """与 ``RecommenderBase.recommend`` 同签名、同返回结构。"""
        if not self.enabled:
            # 关闭重排：原样透传，连数组对象都不重建，保证逐条一致
            return self.model.recommend(
                userid,
                user_items,
                N=N,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=filter_items,
                recalculate_user=recalculate_user,
                items=items,
            )

        if np.isscalar(userid):
            result = self._recommend_one(
                int(userid), user_items, N, filter_already_liked_items,
                filter_items, recalculate_user, items,
            )
            self.last_report = result
            return result.ids, result.scores

        return self._recommend_batch(
            np.asarray(userid), user_items, N, filter_already_liked_items,
            filter_items, recalculate_user, items,
        )

    def recommend_with_report(
        self,
        userid,
        user_items,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
        items=None,
    ):
        """单用户入口，额外返回 :class:`RerankResult` 逐条报告。"""
        result = self._recommend_one(
            int(userid), user_items, N, filter_already_liked_items,
            filter_items, recalculate_user, items,
        )
        self.last_report = result
        return result.ids, result.scores, result

    def _recommend_one(
        self, userid, user_row, N, filter_already_liked_items,
        filter_items, recalculate_user, items,
    ):
        fetch_n = N * self.recall_oversample if self.enabled else N
        raw_ids, raw_scores = self.model.recommend(
            userid,
            user_row,
            N=fetch_n,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filter_items,
            recalculate_user=recalculate_user,
            items=items,
        )
        raw_ids = np.asarray(raw_ids)
        raw_scores = np.asarray(raw_scores)

        backfill_ids = raw_ids[N:] if self.recall_oversample > 1 else None
        backfill_scores = raw_scores[N:] if self.recall_oversample > 1 else None
        result = self.engine.rerank(
            userid,
            raw_ids[:N],
            raw_scores[:N],
            N=N,
            backfill_ids=backfill_ids,
            backfill_scores=backfill_scores,
        )
        # 分数 dtype 与底层召回保持一致
        if result.scores.dtype != raw_scores.dtype:
            result.scores = result.scores.astype(raw_scores.dtype)
        if result.ids.dtype != raw_ids.dtype:
            result.ids = result.ids.astype(raw_ids.dtype)
        return result

    def _recommend_batch(
        self, userids, user_items, N, filter_already_liked_items,
        filter_items, recalculate_user, items,
    ):
        # 同一用户在一次请求里重复出现：只重排一次，结果按用户去重复用
        unique_ids, first_index = np.unique(userids, return_index=True)
        ordered = unique_ids[np.argsort(first_index)]

        results = {}
        for userid in ordered.tolist():
            # user_items 与 userid 等行；取该用户第一次出现时对应的那一行
            row_index = int(np.flatnonzero(userids == userid)[0])
            results[int(userid)] = self._recommend_one(
                int(userid),
                user_items[row_index],
                N,
                filter_already_liked_items,
                filter_items,
                recalculate_user,
                items,
            )

        # 底层批量 topk 始终返回 (rows, N) 矩形，未填满位补零；包装层保持
        # 同一结构约定，只在报告里区分真实展示位与填充位。
        id_dtype = results[int(ordered[0])].ids.dtype
        score_dtype = results[int(ordered[0])].scores.dtype
        out_ids = np.zeros((len(userids), N), dtype=id_dtype)
        out_scores = np.zeros((len(userids), N), dtype=score_dtype)
        reports = []
        for row, userid in enumerate(userids.tolist()):
            result = results[int(userid)]
            length = len(result.ids)
            if length:
                out_ids[row, :length] = result.ids
                out_scores[row, :length] = result.scores
            reports.append(result)

        self.last_report = reports
        return out_ids, out_scores

    # ------------------------------------------------------------------
    # 透传非推荐接口，保证包装对象仍可当作原模型使用
    # ------------------------------------------------------------------
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def similar_users(self, *args, **kwargs):
        return self.model.similar_users(*args, **kwargs)

    def similar_items(self, *args, **kwargs):
        return self.model.similar_items(*args, **kwargs)

    def save(self, file):
        return self.model.save(file)

    @property
    def item_factors(self):
        return self.model.item_factors

    @property
    def user_factors(self):
        return self.model.user_factors
