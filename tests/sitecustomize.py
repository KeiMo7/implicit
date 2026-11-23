"""测试专用 sitecustomize：在解释器启动阶段（早于 pytest 的警告过滤器解析）
安装本仓库优先的 implicit 导入分流。

通过 ``PYTHONPATH=tests`` 引导。系统若已有 sitecustomize，本模块在最后显式
再导入一次，保持其行为不被覆盖。
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_IMPLICIT = REPO_ROOT / "implicit"

# editable 安装指向的 implicit 编译树，作为 Cython .so 的回落位置
_FALLBACK_ROOTS = []
for entry in list(sys.path):
    candidate = Path(entry) / "implicit"
    if entry and candidate.is_dir() and (candidate / "cpu").is_dir() and Path(entry) != REPO_ROOT:
        root = Path(entry)
        if root not in _FALLBACK_ROOTS:
            _FALLBACK_ROOTS.append(root)
for finder in list(sys.meta_path):
    redirect_map = getattr(finder, "known_source_files", None)
    if isinstance(redirect_map, dict):
        for source_path in redirect_map.values():
            parent = Path(source_path)
            for _ in range(4):
                if parent.name == "implicit":
                    root = parent.parent
                    if root not in _FALLBACK_ROOTS and root != REPO_ROOT:
                        _FALLBACK_ROOTS.append(root)
                    break
                parent = parent.parent


def _locate(base_implicit_dir, relative_parts):
    package_init = base_implicit_dir.joinpath(*relative_parts) / "__init__.py"
    if package_init.is_file():
        return package_init, True
    stem = base_implicit_dir.joinpath(*relative_parts)
    for suffix in importlib.machinery.SOURCE_SUFFIXES:
        source = stem.with_suffix(suffix)
        if source.is_file():
            return source, False
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        extension = stem.with_suffix(suffix)
        if extension.is_file():
            return extension, False
    return None, False


class _RepoFirstImplicitFinder:
    """implicit.* 模块：本仓库有 .py 就用本仓库的；纯二进制 .so 回落到编译树。"""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "implicit" and not fullname.startswith("implicit."):
            return None
        relative = fullname.split(".")[1:] if "." in fullname else []

        location, is_package = _locate(REPO_IMPLICIT, relative)
        if location is None:
            for root in _FALLBACK_ROOTS:
                location, is_package = _locate(root / "implicit", relative)
                if location is not None:
                    break
        if location is None:
            return None
        search_locations = [str(location.parent)] if is_package else None
        return importlib.util.spec_from_file_location(
            fullname, location, submodule_search_locations=search_locations
        )


def _install():
    # 摘掉 scikit-build editable 的重定向 finder
    for finder in list(sys.meta_path):
        if finder.__class__.__name__ in {
            "ScikitBuildRedirectingFinder",
            "ScikitBuildInplaceFinder",
        }:
            sys.meta_path.remove(finder)

    # 移除指向其他 implicit 源码树的 sys.path 条目，防止 .so 的 Cython
    # 初始化 import 被这些路径抢先解析到另一棵树
    sys.path[:] = [
        entry
        for entry in sys.path
        if not (
            entry
            and (Path(entry) / "implicit" / "__init__.py").is_file()
            and Path(entry) != REPO_ROOT
        )
    ]

    # 清掉启动期可能已缓存的、属于其他源码树的 implicit 模块
    for name in list(sys.modules):
        if name == "implicit" or name.startswith("implicit."):
            module = sys.modules.get(name)
            module_file = getattr(module, "__file__", None) or ""
            paths = getattr(module, "__path__", None)
            belongs_here = str(REPO_ROOT) in module_file or (
                paths is not None and any(str(REPO_ROOT) in p for p in paths)
            )
            if not belongs_here:
                del sys.modules[name]

    if not any(isinstance(f, _RepoFirstImplicitFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _RepoFirstImplicitFinder())


_install()

# 链式执行系统 sitecustomize（若存在且不是本模块）
try:
    import sitecustomize as _system_sitecustomize  # noqa: F401
except ImportError:
    pass
