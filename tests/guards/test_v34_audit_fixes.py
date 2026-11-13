"""v3.4 审计批次的回归测试。

对应 2026-11-13 第二轮全量核查（计划 §0.12）发现的 3 个缺陷：

1. **依赖漏声明**：`plotly` / `shap` / `psutil` 出现在代码 import 里，
   但 `pyproject.toml` 一个字都没提；`duckdb` 被放进可选组 `p1`，
   而 `--stage all` 的 `warehouse` 阶段**必然**用到它。
   后果与 v3.3 的 `.gitignore` 缺陷同族：本地（已装依赖）全绿，
   `pip install -e . && make all` 直接 ModuleNotFoundError —— **仓库不可复现**。
2. **质量报告「字段缺失率」恒为空表**：缺失在原始数据里是**哨兵值**
   （`member_id = -1`、字符串空串 / `unknown`），不是 NaN；
   而 `missing_rates()` 只判 `isna()`，于是这一节永远渲染成空表 ——
   逻辑正确、产出恒空，正是「交付了但没产出」。
3. **CLI 缺 `dashboard` stage**：计划 §3.1 明确列了 15 个 stage，
   实现里只有 14 个，看板只能手工 `streamlit run`。
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from retailinsight.etl.quality import _missing_mask, missing_rates

ROOT = Path(__file__).resolve().parents[2]
#: 导入名 → 发行包名（少数包两者不同）
_DIST_ALIAS = {
    "sklearn": "scikit_learn",
    "yaml": "pyyaml",
    "pil": "pillow",
    "cv2": "opencv_python",
    "dateutil": "python_dateutil",
}


def _norm(dist: str) -> str:
    """把 `pkg>=1.0` / `pkg[extra]` 归一成 `pkg`。"""
    return (dist.split(">=")[0].split("[")[0].split("==")[0]
            .strip().lower().replace("-", "_"))


def _declared() -> tuple[set[str], set[str]]:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    main = {_norm(d) for d in cfg["project"]["dependencies"]}
    extra: set[str] = set()
    for group in cfg["project"].get("optional-dependencies", {}).values():
        extra |= {_norm(d) for d in group}
    return main, extra


def _imports(subdirs: tuple[str, ...]) -> dict[str, set[str]]:
    """AST 收集 subdirs 下的第三方顶层导入（不执行代码，不受可选依赖影响）。"""
    mods: dict[str, set[str]] = {}
    files = [p for d in subdirs if (ROOT / d).exists()
             for p in (ROOT / d).rglob("*.py") if "__pycache__" not in str(p)]
    for p in files:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for n in names:
                top = n.split(".")[0]
                if top in sys.stdlib_module_names or top == "retailinsight":
                    continue
                mods.setdefault(top, set()).add(str(p.relative_to(ROOT)))
    return mods


# ------------------------------------------------- 缺陷 1：依赖漏声明

#: 唯一允许「不在主依赖里」的导入：lightgbm 缺失时会**显式降级**到 sklearn HistGB，
#: 并在 metrics_churn.json 的 `backend` 字段留痕（不是静默降级）。
_GRACEFUL_OPTIONAL = {"lightgbm"}


def test_retailinsight_imports_are_all_declared_as_main_dependencies():
    """`retailinsight/` 是包本体：它用到的第三方包必须在**主依赖**里。

    放可选组是不行的 —— `--stage all` 会走完全部 stage，缺任何一个都会崩。
    唯一例外是 lightgbm：它被 `try/except ImportError` 包着且有显式降级留痕。
    """
    main, extra = _declared()
    mods = _imports(("retailinsight",))
    missing = {
        m: sorted(files)[:2]
        for m, files in mods.items()
        if m not in main and _DIST_ALIAS.get(m) not in main
    }
    ungraceful = set(missing) - _GRACEFUL_OPTIONAL
    assert not ungraceful, f"以下导入未声明为主依赖（clone 后必崩）：{ungraceful}"
    for m in _GRACEFUL_OPTIONAL & set(mods):
        assert m in extra | main, f"{m} 既不装也不声明，降级路径无从谈起"


def test_graceful_optional_import_has_visible_fallback():
    """可选依赖的降级必须**可观测**：import 被 try 包着，且 metrics 记 backend。"""
    src = (ROOT / "retailinsight" / "models" / "churn.py").read_text(encoding="utf-8")
    assert "import lightgbm" in src
    assert "try:" in src and "_HAS_LGBM" in src, "lightgbm 必须是 try/except 软依赖"
    assert '"backend"' in src, "缺 lightgbm 时必须把实际后端写进 metrics"


def test_dashboard_imports_are_declared():
    """看板是交付物之一，plotly 必须能装上，否则 streamlit 一启动就 ImportError。"""
    main, extra = _declared()
    missing = {
        m: sorted(files)[:2]
        for m, files in _imports(("dashboard",)).items()
        if m not in main | extra and _DIST_ALIAS.get(m) not in main | extra
    }
    assert not missing, f"看板依赖未声明：{missing}"


def test_scripts_imports_are_declared_in_any_group():
    """脚本层允许用 dev 组依赖（psutil 只服务于压测脚本）。"""
    main, extra = _declared()
    missing = {
        m: sorted(files)[:2]
        for m, files in _imports(("scripts",)).items()
        if m not in main | extra and _DIST_ALIAS.get(m) not in main | extra
    }
    assert not missing, f"脚本依赖未声明：{missing}"


def test_stage_all_required_packages_are_main_dependencies():
    """`--stage all` 必经的三个包（duckdb / plotly / shap）不得只放在可选组。"""
    main, _ = _declared()
    for pkg in ("duckdb", "plotly", "shap"):
        assert pkg in main, f"{pkg} 是 `--stage all` 必经依赖，必须在主依赖里"


def test_makefile_install_covers_all_stages():
    """`make install` 之后必须能直接 `make all`（不含交互式 stage）。"""
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert 'pip install -e ".[dev]"' in text
    main, extra = _declared()
    dev = {_norm(d) for d in tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["optional-dependencies"]["dev"]}
    # dev 组必须自带测试工具，否则 make test 也跑不起来
    assert {"pytest", "pytest_cov", "ruff"} <= dev
    assert extra >= dev


# ------------------------------------------------- 缺陷 2：缺失率恒空

def test_missing_mask_treats_sentinel_as_missing():
    """`-1` 的 ID 与空串/unknown 都算缺失 —— 它们是本项目的缺失表达方式。"""
    s_id = pd.Series([1, 2, -1, -1], name="member_id")
    assert _missing_mask(s_id).sum() == 2

    s_txt = pd.Series(["女", "", "unknown", "未知"], name="gender")
    assert _missing_mask(s_txt).sum() == 3

    s_num = pd.Series([0.0, 1.5, float("nan")], name="amount")
    assert _missing_mask(s_num).sum() == 1


def test_missing_rates_is_not_empty_on_sentinel_only_frame():
    """纯哨兵缺失（零 NaN）也必须统计出来 —— 这正是旧实现给出空表的原因。"""
    df = pd.DataFrame({
        "member_id": [1, 2, 3, -1, -1],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        "gender": ["女", "", "男", "unknown", "男"],
    })
    rates = missing_rates({"transactions": df})
    assert rates["transactions.member_id"] == pytest.approx(0.4)
    assert rates["transactions.gender"] == pytest.approx(0.4)
    assert "transactions.amount" not in rates


def test_quality_report_section3_has_data_rows():
    """产物级断言：报告第三节不能是空表（v3.3 交付时它确实是空的）。"""
    md = ROOT / "reports" / "数据质量报告.md"
    assert md.exists(), "先跑 `make etl` 生成质量报告"
    text = md.read_text(encoding="utf-8")
    head = text.split("## 三、", 1)[1].split("## ", 1)[0]
    rows = [ln for ln in head.splitlines() if ln.startswith("| ") and "|" in ln[2:]]
    # 表头 + 分隔线之后至少要有 1 行真实数据
    assert len(rows) >= 3, f"缺失率表只有表头，没有数据行：\n{head}"
    assert "清洗前缺失率" in head and "清洗后缺失率" in head


# ------------------------------------------------- 缺陷 16：测试污染真实产物

def test_write_quality_report_does_not_touch_real_artifacts(tmp_path):
    """往 tmp 写报告时，**不得**顺手覆盖仓库里的真实产物。

    历史事故（v3.4 查出）：`write_quality_report()` 内部的 HTML 渲染写死了
    `docs/数据质量报告.html`，于是「只想验证 md」的单测把真实 HTML 覆盖成
    `members.x` 这种假数据 —— 与 v3.3 的「测试 unlink 真实产物」同族。
    """
    from retailinsight.etl.quality import build_quality_report, write_quality_report

    real = {p.name: p for p in (ROOT / "reports").glob("数据质量报告.*")}
    stamp = {n: p.read_bytes() for n, p in real.items()}

    rep = build_quality_report({"members": 100}, {"members": 98},
                               {"dedup_removed": 2, "missing_rates": {"members.x": 0.01}})
    write_quality_report(rep, tmp_path / "q.md")

    for name, before in stamp.items():
        assert real[name].read_bytes() == before, (
            f"测试把真实产物 reports/{name} 覆盖了 —— 产物的写入路径必须跟着入参走"
        )


def test_write_quality_report_html_follows_md_path(tmp_path):
    """md / json / html 三份产物应落在同一个目录（同一个入参决定）。"""
    from retailinsight.etl.quality import build_quality_report, write_quality_report

    rep = build_quality_report({"members": 10}, {"members": 9}, {"dedup_removed": 1})
    write_quality_report(rep, tmp_path / "sub" / "q.md")
    assert (tmp_path / "sub" / "q.html").exists()
    assert (tmp_path / "sub" / "数据质量报告.json").exists()


# ------------------------------------------------- 缺陷 3：缺 dashboard stage

def test_dashboard_stage_registered_but_excluded_from_all():
    from retailinsight.cli import ALL_STAGES, INTERACTIVE_STAGES, STAGES

    assert "dashboard" in STAGES
    assert "dashboard" in INTERACTIVE_STAGES
    assert "dashboard" not in ALL_STAGES
    # 其余 14 个 stage 仍必须在 all 里（顺序不能乱）
    assert len(ALL_STAGES) == len(STAGES) - len(INTERACTIVE_STAGES)


def test_dashboard_stage_launches_streamlit(monkeypatch):
    """`--stage dashboard` 应拼出正确的 streamlit 命令（不真的起服务）。"""
    from retailinsight import cli

    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ARG001
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    cli.stage_dashboard(port=8599)

    cmd = captured["cmd"]
    assert "streamlit" in cmd and "run" in cmd
    assert any("dashboard" in str(x) and str(x).endswith("app.py") for x in cmd)
    assert "8599" in cmd
    assert "--server.headless" in cmd
