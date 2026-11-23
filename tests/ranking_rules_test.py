"""规则集加载与启动期校验测试。"""

import json
import os
import tempfile

import pytest

from implicit.ranking import RuleConfigError, load_rules


def _base_config(**overrides):
    config = {
        "groups": [{"name": "series", "attribute": "series"}],
        "rules": [
            {
                "id": "series-cap",
                "type": "group_cap",
                "priority": 50,
                "group": "series",
                "max_items": 2,
                "min_spacing": 1,
                "source": "global-diversity",
            }
        ],
    }
    config.update(overrides)
    return config


def test_load_compiles_immutable_ruleset():
    resolver = load_rules(_base_config())
    ruleset = resolver.global_ruleset
    assert ruleset.group_caps[0].rule_id == "series-cap"
    # 规则按优先级降序排好
    priorities = [rule.priority for rule in ruleset.rules]
    assert priorities == sorted(priorities, reverse=True)


def test_duplicate_group_name_fails_at_startup():
    config = {
        "groups": [
            {"name": "series", "attribute": "series"},
            {"name": "series", "attribute": "line"},
        ],
        "rules": [],
    }
    with pytest.raises(RuleConfigError, match="分组名重复"):
        load_rules(config)


def test_duplicate_rule_id_fails_at_startup():
    config = _base_config()
    config["rules"].append(
        {"id": "series-cap", "type": "blacklist", "priority": 10, "item_ids": [1]}
    )
    with pytest.raises(RuleConfigError, match="规则 id 重复"):
        load_rules(config)


def test_min_greater_than_max_fails_at_startup():
    config = _base_config()
    config["rules"][0]["min_items"] = 5
    config["rules"][0]["max_items"] = 2
    with pytest.raises(RuleConfigError, match="上下限矛盾"):
        load_rules(config)


def test_missing_priority_fails_at_startup():
    config = _base_config()
    del config["rules"][0]["priority"]
    with pytest.raises(RuleConfigError, match="priority"):
        load_rules(config)


def test_unknown_rule_type_fails():
    config = _base_config()
    config["rules"][0]["type"] = "teleport"
    with pytest.raises(RuleConfigError, match="类型未知"):
        load_rules(config)


def test_group_cap_referencing_undefined_group_fails():
    config = _base_config()
    config["rules"][0]["group"] = "shop"
    with pytest.raises(RuleConfigError, match="未定义的分组"):
        load_rules(config)


def test_pin_position_duplicate_fails():
    config = _base_config()
    config["rules"].append(
        {"id": "pin-bad", "type": "pin", "priority": 90,
         "item_ids": [1, 2], "positions": [0, 0]}
    )
    with pytest.raises(RuleConfigError, match="positions 存在重复"):
        load_rules(config)


def test_pin_length_mismatch_fails():
    config = _base_config()
    config["rules"].append(
        {"id": "pin-bad", "type": "pin", "priority": 90, "item_ids": [1], "positions": [0, 1]}
    )
    with pytest.raises(RuleConfigError, match="长度必须一致"):
        load_rules(config)


def test_blacklist_requires_nonempty_items():
    config = _base_config()
    config["rules"].append(
        {"id": "bl-empty", "type": "blacklist", "priority": 80, "item_ids": []}
    )
    with pytest.raises(RuleConfigError):
        load_rules(config)


def test_invalid_max_items_fails():
    config = _base_config()
    config["rules"][0]["max_items"] = 0
    with pytest.raises(RuleConfigError, match="max_items"):
        load_rules(config)


def test_negative_spacing_fails():
    config = _base_config()
    config["rules"][0]["min_spacing"] = -1
    with pytest.raises(RuleConfigError, match="min_spacing"):
        load_rules(config)


def test_user_override_takes_effect_and_missing_user_falls_back():
    config = _base_config()
    config["users"] = {
        "7": {
            "rules": [
                {"id": "user7-cap", "type": "group_cap", "priority": 50,
                 "group": "series", "max_items": 1}
            ]
        }
    }
    resolver = load_rules(config)
    ruleset, fallback = resolver.resolve(7)
    assert fallback is None
    assert ruleset.group_caps[0].max_items == 1

    # 缺省用户回落全局并留痕，不允许静默
    global_ruleset, fallback = resolver.resolve(999)
    assert global_ruleset is resolver.global_ruleset
    assert fallback["fallback"] == "global"
    assert "not found" in fallback["reason"]
    assert fallback["userid"] == 999


def test_corrupt_user_config_falls_back_with_trace():
    config = _base_config()
    config["users"] = {
        # 空规则集：不允许静默使用空规则
        "8": {"rules": []},
        # 非法结构
        "9": {"rules": [{"id": "x", "type": "group_cap", "priority": 1}]},
    }
    resolver = load_rules(config)  # 启动不报错
    assert set(resolver.user_errors) == {"8", "9"}
    ruleset, fallback = resolver.resolve(9)
    assert ruleset is resolver.global_ruleset
    assert fallback["userid"] == 9 and fallback["fallback"] == "global"


def test_load_from_json_file_once():
    config = _base_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "rules.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        resolver = load_rules(path)
        assert resolver.config_source == path
        assert resolver.global_ruleset.group_caps[0].max_items == 2


def test_unreadable_json_fails_at_startup():
    with pytest.raises(RuleConfigError, match="无法读取"):
        load_rules("/nonexistent/path/rules.json")


def test_ruleset_is_frozen_at_load_time():
    config = _base_config()
    config["rules"].append(
        {"id": "bl", "type": "blacklist", "priority": 100, "item_ids": [1]}
    )
    resolver = load_rules(config)
    # RuleSet 用 __slots__ 固化，不允许新增/改写属性
    with pytest.raises(AttributeError):
        resolver.global_ruleset.new_attribute = "x"
    # 规则序列是 tuple，不允许变更
    with pytest.raises(AttributeError):
        resolver.global_ruleset.rules.append("x")
    # 黑名单集合也是冻结的 frozenset
    with pytest.raises(AttributeError):
        resolver.global_ruleset.blacklist[0].item_ids.add(123)
