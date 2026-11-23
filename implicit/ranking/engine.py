"""确定性展示重排引擎。

引擎只消费召回产出的 ``(ids, scores)`` 与启动期冻结的规则集，做四件事：
黑名单剔除、强制保留（pin）、分组限流与打散、不足位回填。模型分数既不会被
修改也不会参与重排决策——顺序只由召回名次与规则决定，因此同一输入永远得到
逐条一致的输出。

复杂度上界与规则数量无关：黑名单用集合判定，每个物品适用的 group_cap 在
请求内预先解析成 O(分组维度) 的查表，放置阶段最坏为 ``O(N^2 * G)``（N 为
展示位数量、G 为分组维度数），不随规则条数增长。
"""

from .rules import UNGROUPED


class ItemCatalog:
    """物品属性目录，在引擎构造时冻结。

    Parameters
    ----------
    attributes : dict 或 callable
        ``{item_id: {attribute_name: value}}`` 的映射（会被复制，后续外部
        改动不影响引擎），或一个 ``item_id -> dict`` 的可调用对象。可调用
        形式必须是确定性的纯查询，不允许在里面读规则配置。
    """

    __slots__ = ("_mapping", "_lookup")

    def __init__(self, attributes=None):
        if attributes is None:
            self._mapping = {}
            self._lookup = None
        elif callable(attributes):
            self._lookup = attributes
            self._mapping = None
        elif isinstance(attributes, dict):
            self._mapping = dict(attributes)
            self._lookup = None
        else:
            raise TypeError("attributes 必须是 dict 或 callable")

    def attributes_of(self, item_id):
        if self._lookup is not None:
            result = self._lookup(item_id)
            return result if isinstance(result, dict) else {}
        return self._mapping.get(item_id, {})


class RerankResult:
    """一次重排的完整结果。

    ``positions`` 与最终 ``ids`` 同源同序、逐条对应；``removed`` 记录每个被
    剔除物品的原因与命中规则，报告与列表不可能对不上。

    Attributes
    ----------
    ids : np.ndarray
    scores : np.ndarray
        与 ids 等长，分数值与 dtype 均沿用召回输出，未做任何修改。
    positions : list[dict]
        每个展示位一条：位置、重排前/后名次、命中规则、是否 pin/回填。
    removed : list[dict]
        每个被剔除物品一条：原名次、原因、规则 id、规则来源。
    conflicts : list[dict]
        规则冲突裁决过程。
    unfulfilled : list[dict]
        无法满足的 pin / min_items 诉求。
    fallback : dict 或 None
        用户规则集回落全局时的留痕。
    userid : int 或 None
    """

    def __init__(self, ids, scores, positions, removed, conflicts, unfulfilled, fallback, userid):
        self.ids = ids
        self.scores = scores
        self.positions = positions
        self.removed = removed
        self.conflicts = conflicts
        self.unfulfilled = unfulfilled
        self.fallback = fallback
        self.userid = userid

    @property
    def is_empty(self):
        return len(self.ids) == 0

    def to_dict(self):
        """转成可 JSON 序列化的 dict（numpy 标量转原生类型）。"""
        return {
            "userid": self.userid,
            "ids": [int(i) for i in self.ids],
            "scores": [float(s) for s in self.scores],
            "positions": self.positions,
            "removed": self.removed,
            "conflicts": self.conflicts,
            "unfulfilled": self.unfulfilled,
            "fallback": self.fallback,
        }


# slot 元组结构（7 字段，落位后不可变）:
# (item_id, score, original_rank, pin_rule, is_backfill, source, relaxed_spacing)
def _make_slot(item_id, score, original_rank, pin_rule=None,
               is_backfill=False, source=None, relaxed=False):
    return (item_id, score, original_rank, pin_rule, is_backfill, source, relaxed)


class RerankEngine:
    """重排引擎。规则解析器与物品目录均在构造时注入，运行期无配置 I/O。"""

    def __init__(self, resolver, catalog=None):
        self.resolver = resolver
        self.catalog = catalog if catalog is not None else ItemCatalog()

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def rerank(
        self,
        userid,
        ids,
        scores,
        N=10,
        backfill_ids=None,
        backfill_scores=None,
        backfill_source="recall-ranked-beyond-N",
    ):
        """对单个用户的召回结果重排。

        Parameters
        ----------
        userid : int 或 None
            用于选择规则集；None 时直接使用全局规则集。
        ids, scores : array_like
            召回输出，按召回名次降序。分数只透传，不参与决策。
        N : int
            目标展示位数。
        backfill_ids, backfill_scores : array_like, 可选
            回填候选池（如召回扩大到 N 之外的部分），按优先级排序。
        backfill_source : str
            回填来源标识，写入报告保证可追溯。
        """
        import numpy as np

        ids = np.asarray(ids)
        scores = np.asarray(scores)
        if ids.shape != scores.shape or ids.ndim != 1:
            raise ValueError("ids 与 scores 必须是等长的一维数组")
        if N < 0:
            raise ValueError("N 必须 >= 0")

        ruleset, fallback = self._resolve(userid)
        conflicts = []
        removed = []
        unfulfilled = []

        # 每个分组维度实际生效的 cap：同组多条 group_cap 时按优先级裁决一次
        effective_caps = self._effective_group_caps(ruleset, conflicts)

        # 规则全集黑名单：主候选与回填池统一用它判定，确保黑名单物品无法
        # 经任何路径进入展示
        full_blacklist = set()
        for rule in ruleset.blacklist:
            full_blacklist.update(rule.item_ids)

        # 1) 规范化全部候选。主候选 source=None；回填池带来源标识。
        #    全程保留 numpy 标量，避免 float32 -> float64 -> float32 精度抖动
        seen = set()
        entries = []  # (original_rank, item_id, score, source)
        for rank, (item_id, score) in enumerate(zip(ids.tolist(), scores)):
            item_id = int(item_id)
            if item_id in seen:
                removed.append(
                    {"item_id": item_id, "original_rank": rank,
                     "reason": "duplicate", "rule_id": None, "source": None}
                )
                continue
            seen.add(item_id)
            entries.append((rank, item_id, score, None))

        backfill_entries = self._prepare_backfill(
            backfill_ids, backfill_scores, seen, full_blacklist, len(ids),
        )
        # 回填池里的黑名单/重复物品在 _prepare_backfill 已挡掉；这里把合格
        # 回填物品也纳入统一候选，并补充分组信息
        groups_of = {item_id: self._groups_of(ruleset, item_id)
                     for _, item_id, _, _ in entries}
        for rank, item_id, score, source in backfill_entries:
            entries.append((rank, item_id, score, source))
            groups_of[item_id] = self._groups_of(ruleset, item_id)

        # 2) 黑名单 / pin 裁决（基于合并后的完整候选，pin 可命中回填池物品）
        blacklisted = self._adjudicate_claims(ruleset, entries, conflicts, removed)

        # 3) pin 占位（被 pin 的回填物品会被捞回锁定，并保留其回填来源）
        slots, pinned_items, unfulfilled_pins = self._place_pins(
            ruleset, entries, blacklisted, N, conflicts
        )
        unfulfilled.extend(unfulfilled_pins)

        # 分组状态在每次落位时即时维护，spacing 判断始终基于当前已落位物品。
        # 统计按 (分组名, 分组值) 复合键，规则按分组名查，二者不能混用。
        group_count = {}
        last_position = {}

        def register(item_id, position):
            for group_name, group_value in groups_of.get(item_id, {}).items():
                if group_value is UNGROUPED:
                    continue
                key = (group_name, group_value)
                group_count[key] = group_count.get(key, 0) + 1
                last_position[key] = position

        for position, slot in enumerate(slots):
            if slot is not None:
                register(slot[0], position)

        def cap_allows(item_id):
            for group_name, group_value in groups_of[item_id].items():
                if group_value is UNGROUPED:
                    continue
                rule = effective_caps.get(group_name)
                if rule is None:
                    continue
                if group_count.get((group_name, group_value), 0) >= rule.max_items:
                    return False
            return True

        def spacing_allows(item_id, position, strict):
            if not strict:
                return True
            for group_name, group_value in groups_of[item_id].items():
                if group_value is UNGROUPED:
                    continue
                rule = effective_caps.get(group_name)
                if rule is None or rule.min_spacing <= 0:
                    continue
                key = (group_name, group_value)
                if key in last_position and position - last_position[key] <= rule.min_spacing:
                    return False
            return True

        def place_from(pool, position, strict):
            for index, entry in enumerate(pool):
                rank, item_id, score, source = entry
                if cap_allows(item_id) and spacing_allows(item_id, position, strict):
                    pool.pop(index)
                    slots[position] = _make_slot(
                        item_id, score, rank,
                        is_backfill=source is not None,
                        source=source,
                        relaxed=not strict,
                    )
                    register(item_id, position)
                    return True
            return False

        # 4) 统一候选池：主候选在前（按召回名次），回填候选在后。
        #    先严格（cap+spacing 全局生效）后放宽（只保留 cap 硬上限），
        #    两个阶段都跨主候选/回填候选统一调度，保证打散约束对回填物品同样成立。
        pool = [e for e in entries if e[1] not in blacklisted and e[1] not in pinned_items]
        for strict in (True, False):
            for position in range(N):
                if slots[position] is None:
                    place_from(pool, position, strict)

        # 5) 未放入的候选逐一记录原因（cap 挡住或没有空位），来源可追溯
        for rank, item_id, _score, source in pool:
            rule = self._first_violated_cap(effective_caps, groups_of[item_id], group_count)
            reason = "backfill_not_placed" if source is not None else "group_cap_exhausted"
            if rule is None and source is None:
                # 主候选因 spacing 放宽后仍无空位等情况，归入 cap 类剔除留痕
                reason = "not_placed"
            removed.append(
                {"item_id": item_id, "original_rank": rank,
                 "reason": reason,
                 "rule_id": rule.rule_id if rule else None,
                 "source": source if source is not None else (rule.source if rule else None)}
            )

        # 6) 候选耗尽仍有空位：压缩，pin 与相对次序保持不变
        compacted = any(slot is None for slot in slots)
        placed = [slot for slot in slots if slot is not None]

        self._record_unmet_min_items(effective_caps, group_count, unfulfilled)
        positions = self._build_positions(placed, groups_of, effective_caps, compacted)

        out_ids = np.array([slot[0] for slot in placed], dtype=ids.dtype)
        out_scores = np.array([slot[1] for slot in placed], dtype=scores.dtype)

        return RerankResult(
            ids=out_ids,
            scores=out_scores,
            positions=positions,
            removed=removed,
            conflicts=conflicts,
            unfulfilled=unfulfilled,
            fallback=fallback,
            userid=userid,
        )

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _resolve(self, userid):
        if userid is None:
            return self.resolver.global_ruleset, None
        return self.resolver.resolve(userid)

    def _groups_of(self, ruleset, item_id):
        attrs = self.catalog.attributes_of(item_id)
        result = {}
        for group in ruleset.groups:
            value = attrs.get(group.attribute)
            result[group.name] = value if value is not None else UNGROUPED
        return result

    @staticmethod
    def _effective_group_caps(ruleset, conflicts):
        """每个分组只保留优先级最高的 group_cap；同组多规则记一次裁决。"""
        winners = {}
        for rule in ruleset.group_caps:
            current = winners.get(rule.group)
            if current is None:
                winners[rule.group] = rule
                continue
            if rule.priority > current.priority:
                winner, loser = rule, current
            else:
                winner, loser = current, rule
            winners[rule.group] = winner
            conflicts.append(
                {"type": "group_cap_conflict", "group": rule.group,
                 "winner_rule_id": winner.rule_id, "loser_rule_id": loser.rule_id,
                 "decision": "higher explicit priority wins"}
            )
        return winners

    @staticmethod
    def _adjudicate_claims(ruleset, entries, conflicts, removed):
        """对每个物品汇总黑名单/pin 诉求，按显式优先级裁决，返回黑名单集合。"""
        blacklisted = set()
        item_set = {item_id for _, item_id, _, _ in entries}

        black_hits = {}
        for rule in ruleset.blacklist:
            for item_id in rule.item_ids:
                if item_id in item_set:
                    black_hits.setdefault(item_id, []).append(rule)

        pin_hits = {}
        for rule in ruleset.pins:
            for item_id in rule.item_ids:
                if item_id in item_set:
                    pin_hits.setdefault(item_id, []).append(rule)

        rank_of = {item_id: rank for rank, item_id, _, _ in entries}
        for item_id, black_rules in black_hits.items():
            top_black = black_rules[0]  # ruleset.rules 已按优先级排序
            pin_rules = pin_hits.get(item_id, [])
            top_pin = pin_rules[0] if pin_rules else None
            if top_pin is not None and top_pin.priority > top_black.priority:
                for loser in black_rules:
                    conflicts.append(
                        {"type": "blacklist_vs_pin", "item_id": item_id,
                         "winner_rule_id": top_pin.rule_id, "loser_rule_id": loser.rule_id,
                         "decision": "higher explicit priority wins, item kept and pinned"}
                    )
            else:
                for pin in pin_rules:
                    conflicts.append(
                        {"type": "blacklist_vs_pin", "item_id": item_id,
                         "winner_rule_id": top_black.rule_id, "loser_rule_id": pin.rule_id,
                         "decision": "higher explicit priority wins, item removed"}
                    )
                blacklisted.add(item_id)
                removed.append(
                    {"item_id": item_id, "original_rank": rank_of[item_id],
                     "reason": "blacklist", "rule_id": top_black.rule_id,
                     "source": top_black.source}
                )
        return blacklisted

    @staticmethod
    def _place_pins(ruleset, entries, blacklisted, N, conflicts):
        """处理 pin：位置冲突按优先级裁决。返回 slots、被 pin 物品、未竟诉求。

        被 pin 的物品即使只存在于回填池，也会被捞回锁定；其回填来源保留在
        slot 上，报告里同时标记 pinned 与 backfilled。
        """
        slots = [None] * N
        pinned_items = set()
        unfulfilled = []
        rank_of = {item_id: rank for rank, item_id, _, _ in entries}
        score_of = {item_id: score for _, item_id, score, _ in entries}
        source_of = {item_id: source for _, item_id, _, source in entries}

        # 每个物品只接受最高优先级的 pin 规则；同物品多规则记裁决
        owner = {}
        owner_position = {}
        for rule in ruleset.pins:
            for item_id, position in zip(rule.item_ids, rule.positions):
                if item_id not in rank_of or item_id in blacklisted:
                    continue
                current = owner.get(item_id)
                if current is None:
                    owner[item_id] = rule
                    owner_position[item_id] = position
                else:
                    conflicts.append(
                        {"type": "pin_rule_conflict", "item_id": item_id,
                         "winner_rule_id": current.rule_id, "loser_rule_id": rule.rule_id,
                         "decision": "higher explicit priority wins, duplicate pin ignored"}
                    )

        # 按优先级（再按 item_id 保证确定性）抢占位置
        for item_id, rule in sorted(
            owner.items(), key=lambda kv: (-kv[1].priority, kv[1].rule_id, kv[0])
        ):
            position = owner_position[item_id]
            if position >= N:
                unfulfilled.append(
                    {"type": "pin_out_of_range", "item_id": item_id,
                     "rule_id": rule.rule_id, "source": rule.source,
                     "position": position, "display_size": N}
                )
                continue
            if slots[position] is not None:
                incumbent = slots[position][3]
                conflicts.append(
                    {"type": "pin_position_conflict", "position": position, "item_id": item_id,
                     "winner_rule_id": incumbent.rule_id, "loser_rule_id": rule.rule_id,
                     "decision": "position already claimed by higher priority pin, item demoted"}
                )
                continue
            source = source_of[item_id]
            slots[position] = _make_slot(
                item_id, score_of[item_id], rank_of[item_id], pin_rule=rule,
                is_backfill=source is not None, source=source,
            )
            pinned_items.add(item_id)

        # pin 了但召回里没有（或已被更高优先级黑名单剔除）的物品
        for rule in ruleset.pins:
            for item_id in rule.item_ids:
                if item_id in blacklisted:
                    continue
                if item_id not in rank_of:
                    unfulfilled.append(
                        {"type": "pin_item_not_in_recall", "item_id": item_id,
                         "rule_id": rule.rule_id, "source": rule.source}
                    )

        return slots, pinned_items, unfulfilled

    def _prepare_backfill(self, backfill_ids, backfill_scores, seen,
                          blacklisted, recall_len):
        """规范化回填池：去重、挡黑名单。返回 (rank,id,score,source) 四元组。"""
        if backfill_ids is None:
            return []
        import numpy as np

        backfill_ids = np.asarray(backfill_ids)
        backfill_scores = (
            np.asarray(backfill_scores) if backfill_scores is not None else None
        )
        pool = []
        local_seen = set(seen)
        for pool_index, raw_id in enumerate(backfill_ids.tolist()):
            item_id = int(raw_id)
            if item_id in local_seen or item_id in blacklisted:
                # 重复或黑名单物品绝不允许经回填回到展示
                continue
            local_seen.add(item_id)
            score = (
                backfill_scores[pool_index]
                if backfill_scores is not None and pool_index < len(backfill_scores)
                else float("nan")
            )
            pool.append((recall_len + pool_index, item_id, score, "recall-ranked-beyond-N"))
        return pool

    @staticmethod
    def _record_unmet_min_items(effective_caps, group_count, unfulfilled):
        # min_items 按分组维度聚合：该维度下所有已分组物品的总占位数
        dimension_totals = {}
        for (group_name, _group_value), count in group_count.items():
            dimension_totals[group_name] = dimension_totals.get(group_name, 0) + count
        for group_name, rule in effective_caps.items():
            if rule.min_items and dimension_totals.get(group_name, 0) < rule.min_items:
                unfulfilled.append(
                    {"type": "min_items_unmet", "group": group_name,
                     "rule_id": rule.rule_id, "source": rule.source,
                     "expected_min": rule.min_items,
                     "actual": dimension_totals.get(group_name, 0)}
                )

    def _build_positions(self, placed, groups_of, effective_caps, compacted):
        positions = []
        for final_rank, slot in enumerate(placed):
            item_id, _, original_rank, pin_rule, is_backfill, source, relaxed = slot
            hit_rules = []
            reasons = []
            if pin_rule is not None:
                hit_rules.append(pin_rule.rule_id)
                reasons.append("pinned position")
            for group_name, value in groups_of.get(item_id, {}).items():
                if value is UNGROUPED:
                    continue
                rule = effective_caps.get(group_name)
                if rule is not None:
                    hit_rules.append(rule.rule_id)
            if is_backfill:
                reasons.append("backfilled from recall-ranked-beyond-N")
            if relaxed:
                reasons.append("spacing relaxed to fill slot")
            if compacted:
                reasons.append("empty slots compacted due to insufficient candidates")
            positions.append(
                {"position": final_rank, "item_id": item_id,
                 "original_rank": original_rank, "final_rank": final_rank,
                 "rules": hit_rules, "reasons": reasons,
                 "pinned": pin_rule is not None,
                 "backfilled": is_backfill,
                 "backfill_source": source if is_backfill else None}
            )
        return positions

    @staticmethod
    def _first_violated_cap(effective_caps, item_groups, group_count):
        for group_name, group_value in item_groups.items():
            if group_value is UNGROUPED:
                continue
            rule = effective_caps.get(group_name)
            if rule is None:
                continue
            if group_count.get((group_name, group_value), 0) >= rule.max_items:
                return rule
        return None
