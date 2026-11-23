# 展示重排层验收报告

本报告对照需求逐条说明实现位置、测试证据与产物。所有测试均可离线运行，
不引入任何新依赖（仅用标准库 + 已有的 numpy/scipy）。

## 1. 交付物

| 文件 | 作用 |
| --- | --- |
| `implicit/ranking/rules.py` | 规则数据模型、JSON/dict 加载与**启动期一次校验**，编译为不可变 `RuleSet` |
| `implicit/ranking/engine.py` | 确定性重排引擎：限流、打散、黑名单、强制保留、回填、冲突裁决、逐条报告 |
| `implicit/ranking/reranking.py` | `RerankingRecommender` 独立包装层，`recommend` 签名与返回结构不变 |
| `implicit/ranking/__init__.py` | 公共 API 导出 |
| `tests/ranking_rules_test.py` | 配置加载、非法配置启动报错、用户回落留痕（17 项） |
| `tests/ranking_engine_test.py` | 引擎全部语义（22 项） |
| `tests/ranking_reranking_test.py` | 包装层接口兼容、开关透传、单/批量路径（11 项） |
| `tests/ranking_performance_test.py` | 耗时上界、不随规则数线性失控（2 项） |
| `examples/reranking_demo.py` | 同系列扎堆场景端到端演示，产出逐条报告 |
| `tests/sitecustomize.py`、`tests/conftest.py`、`run_tests.sh` | 本仓库无 .so 产物时的导入分流与标准测试入口 |

## 2. 需求逐条对照

### 一、需要覆盖的形态

| # | 需求 | 实现 | 测试 |
| --- | --- | --- | --- |
| 1 | 按属性分组限流，同系列/同店铺最多 N 个位置 | `GroupCapRule.max_items`，多维度独立统计，复合键 `(分组名, 分组值)` | `test_group_cap_limits_series_occupancy`、`test_multi_dimensional_groups_series_and_shop` |
| 2 | 黑名单/强制保留，记录规则来源 | `BlacklistRule`/`PinRule` 均带 `source`，剔除与命中全程留痕 | `test_blacklist_removes_items_and_records_source`、`test_pin_locks_position...` |
| 3 | 同组条目至少间隔 M 个位置 | `GroupCapRule.min_spacing`，严格阶段全局生效；候选不足时放宽并逐条标记 `spacing relaxed` | `test_spacing_enforces_min_distance`、`test_spacing_relaxation_is_recorded_not_silent` |
| 4 | 数量不足按规则回填，来源可追溯 | 回填池统一过黑名单/去重/限流，报告标记 `backfilled` 与 `backfill_source` | `test_backfill_fills_empty_slots_with_traceable_source` |
| 5 | 每个位置给出重排前后名次、命中规则、剔除原因 | `RerankResult.positions`（original_rank/final_rank/rules/reasons）与 `removed` 同源 | `test_report_and_final_list_are_same_source` |
| 6 | 按用户维度配置规则集，缺省走全局 | `RuleSetResolver.resolve(userid)` | `test_user_override_takes_effect_and_missing_user_falls_back` |
| 7 | 规则冲突按显式优先级裁决并记录 | 所有规则带整数 `priority`，裁决写入 `conflicts` | `test_blacklist_vs_pin_conflict...`、`test_duplicate_group_caps...`、`test_pin_position_conflict...` |
| 8 | 同一输入重复重排逐条一致 | 决策只依赖召回名次与冻结规则，无随机/时间因素 | `test_rerank_is_deterministic_across_repeated_calls` |
| 9 | 关闭重排输出与现在逐条一致 | `enabled=False` 直接透传底层数组 | `test_disabled_rerank_is_identical_to_raw_model` |
| 10 | 同组在不同用户规则下上限不同；解析失败回落全局并留痕 | 用户覆盖 `RuleSet`；损坏配置进 `user_errors`，请求时回落并填 `fallback` | `test_same_group_has_different_cap_per_user`、`test_corrupt_user_ruleset_falls_back_with_record` |
| 11 | 重排层可独立替换 | 引擎构造注入，`wrapped.engine = ...` 即可替换；包装基于鸭子类型，不限定召回来源 | `test_engine_is_replaceable` |
| 12 | 耗时相对召回有上界，不随规则数线性失控 | 规则请求内预解析为 O(分组维度) 查表，放置 O(N²·G)，与规则条数无关 | `test_runtime_does_not_grow_linearly_with_rule_count`、`test_rerank_is_cheaper_than_recall...` |

### 二、硬约束

1. **打分不被修改**：分数全程 numpy 标量透传，不参与任何决策；单用户路径输出与底层
   逐位相等（`test_scores_are_never_modified`、`test_no_rules_preserves_single_user_output_bit_exact`），
   模型 `user_factors/item_factors` 在重排前后不变（`test_model_scoring_path_not_touched`）。
2. **记录与列表同源**：`positions` 由最终 `placed` 列表直接构造，逐条断言可对上
   （`test_report_and_final_list_are_same_source`）。
3. **接口调用方式与返回结构不变**：`recommend` 签名完全一致，单用户一维、批量二维；
   过滤语义不变（`test_recommend_signature_and_shapes_unchanged`、`test_existing_filter_semantics_unchanged`）。
4. **不引入新依赖、离线可跑**：仅标准库（json/importlib/pathlib）+ numpy/scipy。
5. **配置启动期一次加载校验**：`load_rules` 返回冻结的 `RuleSetResolver`，运行期无配置 I/O；
   `RuleSet`/规则对象用 `__slots__` + tuple/frozenset 固化（`test_ruleset_is_frozen_at_load_time`）。

### 三、失败与降级

1. **非法配置启动即报错**：分组名重复、规则 id 重复、上下限矛盾、未知类型、缺 priority
   等均抛 `RuleConfigError`（`tests/ranking_rules_test.py` 多个用例）。
2. **全部剔除返回空并说明原因**：返回空数组且 `removed` 逐条说明，黑名单物品无法经回填混入
   （`test_all_blacklisted_returns_empty_with_reason`、`test_blacklisted_item_never_reenters_via_backfill`）。
3. **用户配置缺失/损坏回落全局并留痕**：不静默使用空规则，空规则集本身也被判为损坏
   （`test_corrupt_user_config_falls_back_with_record`）。
4. **同请求重复用户去重**：批量路径按用户缓存结果复用，不重复占展示位
   （`test_duplicate_users_in_batch_are_deduped`）；召回列表内重复物品同样去重
   （`test_duplicate_items_in_request_deduped`）。

### 四、不破坏的现有行为

- 过滤语义与返回类型不变（同上接口测试）。
- 单用户与批量两条路径均可用，且结果一致（`test_batch_and_single_user_paths_agree`）。
- 模型推理/打分路径零改动：包装层只读取 `recommend` 输出。
- 既有推荐测试全部通过：**164 passed, 7 skipped**（GPU 用例，本机无 CUDA），
  见 `reports/original_recommendation_tests.txt`。

### 五、测试与验收

| 验收点 | 对应用例 |
| --- | --- |
| 同系列扎堆 → 限流/打散位置分布 | `test_group_cap_limits_series_occupancy`、`test_spacing_enforces_min_distance` |
| 黑名单剔除、强制保留、回填来源 | `test_blacklist_removes...`、`test_pin_locks...`、`test_backfill_fills...`、`test_pin_can_recover_item_from_backfill_pool` |
| 冲突裁决与非法配置报错 | `test_*conflict*`、`tests/ranking_rules_test.py` 全部 `test_*fails*` |
| 关闭重排输出一致 | `test_disabled_rerank_is_identical_to_raw_model` |
| 原有推荐测试 | `reports/original_recommendation_tests.txt`（164 passed, 7 skipped） |

## 3. 测试结果

```
全量：219 passed, 7 skipped（见 reports/full_test_output_summary.txt）
其中原有推荐相关：164 passed, 7 skipped（见 reports/original_tests_summary.txt）
新增重排：55 passed
```

运行方式：

```bash
./run_tests.sh                                   # 全量（除 GPU）
PYTHONPATH=tests python -m pytest tests/ranking_engine_test.py   # 指定模块
PYTHONPATH=tests python examples/reranking_demo.py               # 端到端演示
```

> 关于 `PYTHONPATH=tests`：本机 implicit 是指向另一棵源码树（含已编译 .so）的
> editable 安装，而本仓库是无 .so 的纯源码树。`tests/sitecustomize.py` 在解释器
> 启动期安装导入分流——纯 Python 模块（含新的 `implicit.ranking`）从本仓库加载，
> Cython 扩展回落到已编译树。这是测试环境引导，不属于库运行时依赖。

## 4. 设计说明与边界

- **硬约束 vs 软约束**：分组占位上限 `max_items` 是任何阶段都不违反的硬上限；
  `min_spacing` 是优先满足的软约束，严格阶段保证，仅当候选多样性不足以填满时
  才在放宽阶段打破并**逐条留痕**，绝不静默。
- **强制保留（pin）与限流的关系**：pin 物品被锁定到指定位置、不会被限流剔除，
  但它**计入所属分组的总占位数**——即 cap 约束的是该分组在整个列表中的占位总量。
  因此当 pin 已占满某分组上限时，普通候选不再向该分组放行（见
  `test_pin_overrides_group_cap_force_keep_semantics`）。pin 位置本身不可移动，
  故当两条 pin 规则把同组物品锁到相邻位时，打散间距让位于强制保留（pin 优先级
  更高且位置是运营显式指定的）。
- **回填与 pin 的关系**：被强制保留的物品若只出现在回填池，会被捞回锁定位置，
  报告同时标记 `pinned` 与 `backfilled`，来源可追溯。
- **用户规则集语义**：用户覆盖是完整替换而非合并（便于独立裁决与留痕），需要的
  全局规则需在用户配置中重声明；缺失或损坏时回落全局并在 `fallback` 字段记录原因。
- **批量结果形状**：沿用底层固定矩形约定 `(rows, N)`，不足位补零；单用户返回短列表，
  与底层 `topk` 行为一致。逐位有效长度以 `last_report` 为准。
