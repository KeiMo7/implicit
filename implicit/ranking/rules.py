"""规则集的数据模型、加载与启动期校验。

配置在启动期一次性加载、校验并编译成不可变的 :class:`RuleSet`，运行期不再读取
任何配置文件或字典。非法的全局配置在加载时直接抛出 :class:`RuleConfigError`，
不带病运行；单个用户的覆盖配置损坏时不影响启动，而是在该用户请求时回落全局
并留下记录（见 :class:`RuleSetResolver`）。

配置 schema（JSON 或等价 dict）::

    {
      "groups": [
        {"name": "series", "attribute": "series"},
        {"name": "shop",   "attribute": "shop"}
      ],
      "rules": [
        {"id": "blacklist-0", "type": "blacklist", "priority": 100,
         "item_ids": [12, 13], "source": "ops-blacklist"},
        {"id": "pin-hero", "type": "pin", "priority": 90,
         "item_ids": [7], "positions": [0], "source": "campaign-42"},
        {"id": "series-cap", "type": "group_cap", "priority": 50,
         "group": "series", "max_items": 2, "min_spacing": 2,
         "min_items": 0, "source": "global-diversity-policy"}
      ],
      "users": {
        "7": {"groups": [...], "rules": [...]}
      }
    }

规则类型：

* ``blacklist``：命中即剔除，``item_ids`` 为物品 id 列表。
* ``pin``：强制保留，``item_ids`` 与 ``positions`` 等长，把指定物品锁定到
  指定位置（位置从 0 开始）。
* ``group_cap``：分组限流与打散，``max_items`` 为同组占位上限，
  ``min_spacing`` 为同组相邻条目之间至少间隔的其他物品数，``min_items``
  为可选的占位下限（仅用于校验与报告，不参与强行凑数）。

所有规则都必须带显式整数 ``priority``，数值越大优先级越高；同优先级按
``id`` 字典序裁决，保证确定性。
"""

import json

# 物品缺少某分组属性时归入的哨兵组值，哨兵组不参与限流/打散
UNGROUPED = None


class RuleConfigError(ValueError):
    """规则配置非法（结构错误、上下限矛盾、名字重复等）。"""


class GroupRule:
    """分组定义：按物品的某个属性（如 series/shop）划分分组。"""

    __slots__ = ("name", "attribute")

    def __init__(self, name, attribute):
        self.name = name
        self.attribute = attribute

    def __repr__(self):
        return f"GroupRule(name={self.name!r}, attribute={self.attribute!r})"


class _Rule:
    __slots__ = ("rule_id", "priority", "source")

    def __init__(self, rule_id, priority, source):
        self.rule_id = rule_id
        self.priority = priority
        self.source = source


class BlacklistRule(_Rule):
    __slots__ = ("item_ids",)

    def __init__(self, rule_id, priority, item_ids, source):
        super().__init__(rule_id, priority, source)
        self.item_ids = frozenset(item_ids)


class PinRule(_Rule):
    """强制保留规则。item_ids[i] 锁定到 positions[i]。"""

    __slots__ = ("item_ids", "positions")

    def __init__(self, rule_id, priority, item_ids, positions, source):
        super().__init__(rule_id, priority, source)
        self.item_ids = tuple(item_ids)
        self.positions = tuple(positions)


class GroupCapRule(_Rule):
    __slots__ = ("group", "max_items", "min_spacing", "min_items")

    def __init__(self, rule_id, priority, group, max_items, min_spacing, min_items, source):
        super().__init__(rule_id, priority, source)
        self.group = group
        self.max_items = max_items
        self.min_spacing = min_spacing
        self.min_items = min_items


class RuleSet:
    """编译后的不可变规则集。

    Attributes
    ----------
    groups : tuple[GroupRule]
        分组定义，按 name 可查。
    rules : tuple
        全部规则，按 priority 降序、id 升序排好，运行期按此顺序裁决。
    blacklist : tuple[BlacklistRule]
    pins : tuple[PinRule]
    group_caps : tuple[GroupCapRule]
    """

    __slots__ = ("name", "groups", "rules", "blacklist", "pins", "group_caps", "_group_by_name")

    def __init__(self, name, groups, rules):
        self.name = name
        self.groups = tuple(groups)
        self.rules = tuple(sorted(rules, key=lambda r: (-r.priority, r.rule_id)))
        self._group_by_name = {g.name: g for g in self.groups}
        self.blacklist = tuple(r for r in self.rules if isinstance(r, BlacklistRule))
        self.pins = tuple(r for r in self.rules if isinstance(r, PinRule))
        self.group_caps = tuple(r for r in self.rules if isinstance(r, GroupCapRule))

    def group_attribute(self, group_name):
        return self._group_by_name[group_name].attribute

    def is_empty(self):
        return not self.rules


def _require(mapping, key, rule_id, kind):
    if key not in mapping:
        raise RuleConfigError(f"{kind} 规则 {rule_id!r} 缺少字段 {key!r}")
    return mapping[key]


def _as_priority(mapping, rule_id):
    if "priority" not in mapping:
        raise RuleConfigError(f"规则 {rule_id!r} 缺少显式 priority")
    priority = mapping["priority"]
    # bool 是 int 的子类，显式排除，避免 True/False 混进来
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise RuleConfigError(f"规则 {rule_id!r} 的 priority 必须是整数")
    return priority


def _as_item_id_list(mapping, key, rule_id):
    value = _require(mapping, key, rule_id, mapping.get("type", "?"))
    if not isinstance(value, list) or not value:
        raise RuleConfigError(f"规则 {rule_id!r} 的 {key} 必须是非空列表")
    normalized = []
    for item_id in value:
        if not isinstance(item_id, int) or isinstance(item_id, bool) or item_id < 0:
            raise RuleConfigError(f"规则 {rule_id!r} 的 {key} 含非法 item id: {item_id!r}")
        normalized.append(item_id)
    return normalized


def _compile_groups(raw_groups, scope):
    if not isinstance(raw_groups, list):
        raise RuleConfigError(f"{scope} 的 groups 必须是列表")
    groups = []
    seen = set()
    for entry in raw_groups:
        if not isinstance(entry, dict):
            raise RuleConfigError(f"{scope} 的 group 定义必须是对象: {entry!r}")
        name = entry.get("name")
        attribute = entry.get("attribute")
        if not isinstance(name, str) or not name:
            raise RuleConfigError(f"{scope} 存在缺少 name 的 group 定义")
        if not isinstance(attribute, str) or not attribute:
            raise RuleConfigError(f"{scope} group {name!r} 缺少 attribute")
        if name in seen:
            raise RuleConfigError(f"{scope} 分组名重复: {name!r}")
        seen.add(name)
        groups.append(GroupRule(name, attribute))
    return groups


def _compile_rules(raw_rules, group_names, scope):
    if not isinstance(raw_rules, list):
        raise RuleConfigError(f"{scope} 的 rules 必须是列表")
    rules = []
    seen_ids = set()
    for entry in raw_rules:
        if not isinstance(entry, dict):
            raise RuleConfigError(f"{scope} 的规则定义必须是对象: {entry!r}")
        rule_id = entry.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            raise RuleConfigError(f"{scope} 存在缺少 id 的规则")
        if rule_id in seen_ids:
            raise RuleConfigError(f"{scope} 规则 id 重复: {rule_id!r}")
        seen_ids.add(rule_id)

        rule_type = entry.get("type")
        priority = _as_priority(entry, rule_id)
        source = entry.get("source", "config")
        if not isinstance(source, str):
            raise RuleConfigError(f"规则 {rule_id!r} 的 source 必须是字符串")

        if rule_type == "blacklist":
            item_ids = _as_item_id_list(entry, "item_ids", rule_id)
            rules.append(BlacklistRule(rule_id, priority, item_ids, source))
        elif rule_type == "pin":
            item_ids = _as_item_id_list(entry, "item_ids", rule_id)
            positions = _as_item_id_list(entry, "positions", rule_id)
            if len(item_ids) != len(positions):
                raise RuleConfigError(
                    f"pin 规则 {rule_id!r} 的 item_ids 与 positions 长度必须一致"
                )
            if len(set(positions)) != len(positions):
                raise RuleConfigError(f"pin 规则 {rule_id!r} 的 positions 存在重复")
            if any(p < 0 for p in positions):
                raise RuleConfigError(f"pin 规则 {rule_id!r} 的 positions 不能为负")
            if len(set(item_ids)) != len(item_ids):
                raise RuleConfigError(f"pin 规则 {rule_id!r} 的 item_ids 存在重复")
            rules.append(PinRule(rule_id, priority, item_ids, positions, source))
        elif rule_type == "group_cap":
            group_name = _require(entry, "group", rule_id, "group_cap")
            if group_name not in group_names:
                raise RuleConfigError(
                    f"group_cap 规则 {rule_id!r} 引用了未定义的分组: {group_name!r}"
                )
            max_items = _require(entry, "max_items", rule_id, "group_cap")
            if not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 1:
                raise RuleConfigError(f"规则 {rule_id!r} 的 max_items 必须是 >=1 的整数")
            min_spacing = entry.get("min_spacing", 0)
            if not isinstance(min_spacing, int) or isinstance(min_spacing, bool) or min_spacing < 0:
                raise RuleConfigError(f"规则 {rule_id!r} 的 min_spacing 必须是 >=0 的整数")
            min_items = entry.get("min_items", 0)
            if not isinstance(min_items, int) or isinstance(min_items, bool) or min_items < 0:
                raise RuleConfigError(f"规则 {rule_id!r} 的 min_items 必须是 >=0 的整数")
            if min_items > max_items:
                raise RuleConfigError(
                    f"规则 {rule_id!r} 上下限矛盾: min_items={min_items} > max_items={max_items}"
                )
            rules.append(
                GroupCapRule(
                    rule_id, priority, group_name, max_items, min_spacing, min_items, source
                )
            )
        else:
            raise RuleConfigError(f"规则 {rule_id!r} 的类型未知: {rule_type!r}")
    return rules


def _compile_ruleset(raw, scope, base_groups=()):
    """编译一段配置（全局或单用户覆盖）。任何非法字段都抛 RuleConfigError。"""
    if not isinstance(raw, dict):
        raise RuleConfigError(f"{scope} 配置必须是对象")

    # 用户覆盖可以在全局分组之上补充/重定义分组
    groups = list(base_groups)
    group_names = {g.name for g in groups}
    if "groups" in raw:
        for group in _compile_groups(raw["groups"], scope):
            if group.name in group_names:
                # 用户覆盖允许重定义同名分组（占位上限可不同），替换属性定义
                groups = [g for g in groups if g.name != group.name]
            groups.append(group)
            group_names.add(group.name)

    rules = _compile_rules(raw.get("rules", []), group_names, scope)
    return RuleSet(scope, groups, rules)


class RuleSetResolver:
    """启动期冻结的规则集解析器：按用户 id 取规则集，缺省/损坏回落全局。

    运行期不再触碰任何文件或原始配置字典，只读取编译好的不可变对象。
    """

    __slots__ = ("global_ruleset", "user_rulesets", "user_errors", "config_source")

    def __init__(self, global_ruleset, user_rulesets=None, user_errors=None, config_source=None):
        self.global_ruleset = global_ruleset
        self.user_rulesets = dict(user_rulesets or {})
        self.user_errors = dict(user_errors or {})
        self.config_source = config_source

    def resolve(self, userid):
        """返回 ``(RuleSet, fallback_info)``。

        fallback_info 为 None 表示命中专属配置；否则是记录回落原因的 dict，
        调用方必须把它写进本次请求的报告，不允许静默使用空规则。
        """
        key = str(userid)
        if key in self.user_errors:
            return self.global_ruleset, {
                "userid": userid,
                "fallback": "global",
                "reason": self.user_errors[key],
            }
        ruleset = self.user_rulesets.get(key)
        if ruleset is None:
            return self.global_ruleset, {
                "userid": userid,
                "fallback": "global",
                "reason": "user ruleset not found",
            }
        return ruleset, None


def load_rules(config):
    """加载并校验规则配置，返回冻结的 :class:`RuleSetResolver`。

    Parameters
    ----------
    config : str 或 dict
        配置文件路径（JSON），或已经解析好的配置 dict。文件只在本函数内读取
        一次，返回后的运行期不再有任何配置 I/O。

    Raises
    ------
    RuleConfigError
        全局配置非法（结构错误、分组名重复、上下限矛盾、规则 id 重复等）。
    """
    config_source = None
    if isinstance(config, str):
        config_source = config
        try:
            with open(config, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RuleConfigError(f"无法读取规则配置 {config!r}: {error}") from error
    elif isinstance(config, dict):
        raw = config
    else:
        raise RuleConfigError("config 必须是 JSON 文件路径或 dict")

    global_ruleset = _compile_ruleset(raw, "global")

    user_rulesets = {}
    user_errors = {}
    raw_users = raw.get("users", {})
    if raw_users is None:
        raw_users = {}
    if not isinstance(raw_users, dict):
        raise RuleConfigError("users 必须是 userid -> 覆盖配置 的映射")

    for user_key, user_raw in raw_users.items():
        try:
            if not isinstance(user_key, str) or not user_key:
                raise RuleConfigError("user id 键必须是非空字符串")
            user_ruleset = _compile_ruleset(user_raw, f"user:{user_key}", global_ruleset.groups)
            if user_ruleset.is_empty():
                raise RuleConfigError("用户规则集为空，不允许静默使用空规则")
            user_rulesets[user_key] = user_ruleset
        except RuleConfigError as error:
            # 单个用户配置损坏不拖垮启动；该用户请求时回落全局并留痕
            user_errors[str(user_key)] = str(error)

    return RuleSetResolver(global_ruleset, user_rulesets, user_errors, config_source)
