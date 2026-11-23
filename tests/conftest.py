"""测试引导。

环境里的 implicit 是指向另一棵源码树（含已编译 Cython 扩展）的 editable 安装，
本仓库是纯源码树、没有 .so 产物。导入分流逻辑（本仓库 .py 优先、.so 回落到
编译树）在 :mod:`sitecustomize` 中实现，以便在 pytest 解析警告过滤器之前就
生效。本 conftest 作为非 PYTHONPATH 入口（如 IDE 直接跑 pytest）的兜底，
确保分流一定被安装。
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import sitecustomize  # noqa: E402,F401  触发/复用 _install()
