"""展示重排层 (presentation reranking layer).

该子包在召回结果与展示之间插入一个独立的重排层，只改变展示顺序与展示集合，
不触碰模型打分路径。模块划分：

* :mod:`implicit.ranking.rules`    规则集的数据模型、加载与启动期校验
* :mod:`implicit.ranking.engine`   确定性重排引擎与逐条命中报告
* :mod:`implicit.ranking.reranking` 适配 ``RecommenderBase.recommend`` 的包装层
"""

from .engine import ItemCatalog, RerankEngine, RerankResult
from .reranking import RerankingRecommender
from .rules import (
    GroupRule,
    RuleConfigError,
    RuleSet,
    RuleSetResolver,
    load_rules,
)

__all__ = [
    "GroupRule",
    "ItemCatalog",
    "RerankEngine",
    "RerankResult",
    "RerankingRecommender",
    "RuleConfigError",
    "RuleSet",
    "RuleSetResolver",
    "load_rules",
]
