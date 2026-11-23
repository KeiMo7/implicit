"""展示重排层端到端验收演示。

构造运营反馈中的场景：首页推荐位被同一系列手机占满。脚本对比重排前后的
位置分布，演示限流、打散、黑名单、强制保留、回填与规则冲突裁决，并把逐条
报告（与最终列表同源）写入 reports/ 目录。

运行（从仓库根目录）：
    PYTHONPATH=tests python examples/reranking_demo.py
"""

import json
import os

import numpy as np

from implicit.ranking import ItemCatalog, RerankEngine, load_rules

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")


def build_synthetic_recall():
    """召回 20 个手机：前 12 个几乎全是 series-A（模拟同系列扎堆）。"""
    # (item_id, series, shop, score)
    items = [
        (101, "A", "shop-1", 9.8),
        (102, "A", "shop-1", 9.7),
        (103, "A", "shop-2", 9.6),
        (104, "A", "shop-2", 9.5),
        (105, "A", "shop-3", 9.4),
        (106, "A", "shop-3", 9.3),
        (107, "A", "shop-1", 9.2),
        (108, "A", "shop-1", 9.1),
        (109, "A", "shop-4", 9.0),
        (110, "A", "shop-4", 8.9),
        (201, "B", "shop-5", 8.8),
        (202, "B", "shop-5", 8.7),
        (203, "B", "shop-6", 8.6),
        (301, "C", "shop-7", 8.5),
        (302, "C", "shop-7", 8.4),
        (303, "C", "shop-8", 8.3),
        (401, "D", "shop-9", 8.2),
        (402, "D", "shop-9", 8.1),
        # 召回扩大到 N 之外的回填候选
        (501, "E", "shop-10", 7.9),
        (502, "E", "shop-10", 7.8),
    ]
    return items


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    items = build_synthetic_recall()
    ids = np.array([it[0] for it in items], dtype=np.int32)
    scores = np.array([it[3] for it in items], dtype=np.float32)
    attributes = {
        it[0]: {"series": it[1], "shop": it[2]} for it in items
    }
    N = 10

    config = {
        "groups": [
            {"name": "series", "attribute": "series"},
            {"name": "shop", "attribute": "shop"},
        ],
        "rules": [
            # 黑名单：下架机型，命中即剔除
            {"id": "ops-blacklist", "type": "blacklist", "priority": 100,
             "item_ids": [109], "source": "ops-ticket-2026-11"},
            # 强制保留：营销活动把 301 锁到首位
            {"id": "campaign-pin", "type": "pin", "priority": 90,
             "item_ids": [301], "positions": [0], "source": "campaign-42"},
            # 同系列最多 3 个位置，同系列相邻至少间隔 2 个位置
            {"id": "series-cap", "type": "group_cap", "priority": 50,
             "group": "series", "max_items": 3, "min_spacing": 2,
             "source": "global-diversity-policy"},
            # 同店铺最多 2 个位置
            {"id": "shop-cap", "type": "group_cap", "priority": 40,
             "group": "shop", "max_items": 2,
             "source": "global-diversity-policy"},
        ],
        "users": {
            # 用户 7 的更严格系列上限：同系列最多 2 个。
            # 注意用户规则集是完整替换而非合并，因此需要的全局规则（黑名单、
            # pin）要在此显式重声明；缺失则回落全局会在报告中留痕。
            "7": {
                "rules": [
                    {"id": "ops-blacklist", "type": "blacklist", "priority": 100,
                     "item_ids": [109], "source": "ops-ticket-2026-11"},
                    {"id": "campaign-pin", "type": "pin", "priority": 90,
                     "item_ids": [301], "positions": [0], "source": "campaign-42"},
                    {"id": "user7-series-cap", "type": "group_cap", "priority": 50,
                     "group": "series", "max_items": 2, "min_spacing": 2},
                    {"id": "user7-shop-cap", "type": "group_cap", "priority": 40,
                     "group": "shop", "max_items": 2},
                ]
            }
        },
    }

    resolver = load_rules(config)
    engine = RerankEngine(resolver, ItemCatalog(attributes))

    # 主候选取前 N，其余作为可追溯回填池
    main_ids, main_scores = ids[:N], scores[:N]
    backfill_ids, backfill_scores = ids[N:], scores[N:]

    result = engine.rerank(
        userid=7,
        ids=main_ids,
        scores=main_scores,
        N=N,
        backfill_ids=backfill_ids,
        backfill_scores=backfill_scores,
    )

    attr_of = attributes
    lines = []
    lines.append("展示重排验收报告（用户 7，专属规则：同系列上限 2）")
    lines.append("=" * 78)
    lines.append("")
    lines.append("【重排前】召回顺序（前 10 位，series-A 扎堆）：")
    for rank in range(N):
        item_id = int(main_ids[rank])
        lines.append(
            f"  名次 {rank:2d}  item={item_id}  series={attr_of[item_id]['series']:<3} "
            f"shop={attr_of[item_id]['shop']:<8} score={main_scores[rank]:.2f}"
        )

    lines.append("")
    lines.append("【重排后】最终展示位：")
    for position in result.positions:
        item_id = position["item_id"]
        tags = []
        if position["pinned"]:
            tags.append("PIN")
        if position["backfilled"]:
            tags.append("BACKFILL")
        lines.append(
            f"  位置 {position['position']:2d}  item={item_id}  "
            f"series={attr_of[item_id]['series']:<3} shop={attr_of[item_id]['shop']:<8} "
            f"原名次={position['original_rank']:2d} 规则={','.join(position['rules']) or '-'} "
            f"{'[' + '/'.join(tags) + ']' if tags else ''}"
        )

    lines.append("")
    lines.append("【剔除明细】")
    for record in result.removed:
        lines.append(f"  item={record['item_id']} 原名次={record['original_rank']} "
                     f"原因={record['reason']} 规则={record['rule_id']} 来源={record['source']}")

    lines.append("")
    lines.append("【规则冲突裁决】")
    lines.append(f"  共 {len(result.conflicts)} 条" if result.conflicts
                 else "  本场景无冲突（黑名单/pin 未命中同一物品）")
    for conflict in result.conflicts:
        lines.append(f"  {conflict}")

    lines.append("")
    lines.append("【未竟诉求】")
    lines.append(f"  共 {len(result.unfulfilled)} 条" if result.unfulfilled else "  无")
    for item in result.unfulfilled:
        lines.append(f"  {item}")

    lines.append("")
    lines.append("【规则集回落】")
    lines.append(f"  {result.fallback if result.fallback else '命中用户 7 专属规则，无回落'}")

    # 约束自检
    series_count, shop_count = {}, {}
    series_positions = {}
    relaxed_pairs = []
    for position in result.positions:
        item_id = position["item_id"]
        s = attr_of[item_id]["series"]
        series_count[s] = series_count.get(s, 0) + 1
        series_positions.setdefault(s, []).append(position["position"])
        shop_count[attr_of[item_id]["shop"]] = (
            shop_count.get(attr_of[item_id]["shop"], 0) + 1
        )
        if "spacing relaxed to fill slot" in position["reasons"]:
            relaxed_pairs.append(item_id)

    # 硬约束：限流上限任何情况下都不得违反
    cap_ok = all(count <= 2 for count in series_count.values()) and all(
        count <= 2 for count in shop_count.values()
    )
    # 打散为软约束：严格阶段优先满足；候选不足无法满足时放宽并逐条留痕
    strict_violations = []
    for series, ps in series_positions.items():
        for earlier, later in zip(ps, ps[1:]):
            if later - earlier <= 2:
                later_item = result.positions[later]["item_id"]
                if later_item not in relaxed_pairs:
                    strict_violations.append((series, earlier, later))

    lines.append("")
    lines.append("【约束自检】")
    lines.append(f"  各系列占位数: {series_count}（用户 7 上限=2）")
    lines.append(f"  各店铺占位数: {shop_count}（上限=2）")
    lines.append(f"  限流硬上限全部满足: {cap_ok}")
    if strict_violations:
        lines.append(f"  打散严格约束被违反（不应出现）: {strict_violations}")
    elif relaxed_pairs:
        lines.append(
            f"  打散: {len(relaxed_pairs)} 个位置因候选不足放宽间距并已留痕 -> {relaxed_pairs}"
        )
    else:
        lines.append("  打散: 同系列相邻间隔全部满足 > 2")
    lines.append(f"  黑名单物品 109 已剔除: {109 not in set(result.ids.tolist())}")
    lines.append(f"  pin 物品 301 位于首位: {result.ids[0] == 301}")

    report_text = "\n".join(lines)
    print(report_text)

    with open(os.path.join(REPORTS_DIR, "reranking_demo_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    with open(os.path.join(REPORTS_DIR, "reranking_demo_report.json"), "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"\n报告已写入 {REPORTS_DIR}/reranking_demo_report.{{txt,json}}")


if __name__ == "__main__":
    main()
