"""导入冒烟测试：确保包内每个模块都能被导入。

背景：曾出现过补丁脚本把某个模块的 import 写成 `from ... import (, ITEMS_DIR`
的语法错误，导致该模块**只在被调用时**才炸（其它模块与测试全绿），
压测时才发现。这里用一条测试把这类问题挡在 CI 里。
"""
import importlib
import pkgutil

import pytest

import retailinsight


def test_all_modules_importable():
    failures = []
    for mod in pkgutil.walk_packages(retailinsight.__path__, "retailinsight."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "以下模块无法导入：\n" + "\n".join(failures)


def test_cli_stages_all_importable():
    """CLI 的每个 stage 依赖的模块都要能导入。"""
    modules = [
        "retailinsight.cli",
        "retailinsight.etl.pipeline",
        "retailinsight.etl.warehouse",
        "retailinsight.features.rfm",
        "retailinsight.features.tags",
        "retailinsight.features.behavior",
        "retailinsight.features.churn_features",
        "retailinsight.features.cluster_features",
        "retailinsight.models.churn",
        "retailinsight.models.cluster",
        "retailinsight.models.cluster_compare",
        "retailinsight.models.clv",
        "retailinsight.models.rules",
        "retailinsight.reports.core_reports",
        "retailinsight.reports.excel_report",
        "retailinsight.reports.profile_cards",
        "retailinsight.reports.render_decision_report",
        "retailinsight.viz.charts",
    ]
    for name in modules:
        assert importlib.import_module(name) is not None


@pytest.mark.parametrize("script", ["generate_synthetic.py", "download_data.py",
                                    "build_notebooks.py", "exp_churn_sensitivity.py",
                                    "exp_scale5.py", "fix_lightgbm_dylib.py"])
def test_scripts_compile(script):
    """脚本必须能编译（语法错误在运行前就该暴露，而不是压测跑到一半才炸）。"""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / script
    assert path.exists(), f"缺少脚本 {script}"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_dashboard_compiles():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    path = root / "dashboard" / "app.py"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
