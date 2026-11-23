#!/usr/bin/env bash
# 本仓库测试的标准入口。
#
# 背景：运行环境里的 implicit 是指向另一棵源码树（含已编译 Cython .so）的
# editable 安装，而本仓库是纯源码树、无 .so 产物。tests/sitecustomize.py 负责
# 导入分流（本仓库的纯 Python 模块优先，Cython 扩展回落到已编译树），它必须
# 在 pytest 解析 pyproject 中的 filterwarnings（会提前 import implicit.utils）
# 之前生效，因此需要用 PYTHONPATH=tests 在解释器启动期加载。
#
# 用法：
#   ./run_tests.sh                      # 跑除 GPU 外的全部测试
#   ./run_tests.sh tests/ranking_engine_test.py   # 跑指定测试
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="tests${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS=1

if [ "$#" -eq 0 ]; then
  set -- tests/ --ignore=tests/gpu_test.py
fi

python -m pytest "$@"
