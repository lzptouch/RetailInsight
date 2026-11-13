"""数据质量报告的 HTML 渲染（自包含，无外部依赖）。

与 md / json 版并存：
- md + json 便于在仓库里 diff 与被程序消费
- HTML 便于直接发给业务方看（内联 CSS + 纯 SVG 图表，不需要联网、不需要 JS）
"""
from __future__ import annotations

from pathlib import Path

from retailinsight.settings import DOCS_DIR

_HTML_CSS = """
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
 margin:32px auto;max-width:1080px;color:#1f2933;line-height:1.6}
h1{font-size:24px;border-bottom:2px solid #2f6f4f;padding-bottom:8px}
h2{font-size:18px;margin-top:28px;color:#2f6f4f}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{border:1px solid #d7dee3;padding:6px 10px;text-align:left}
th{background:#eef4f0}
.cmp{display:flex;align-items:center;gap:12px;margin:6px 0}
.cmp .lbl{width:110px;font-size:13px;color:#52606d}
.cmp .bars{flex:1}
.cmp .row{display:flex;align-items:center;gap:8px;height:22px}
.cmp .tag{width:56px;font-size:12px;color:#7b8794;text-align:right}
.cmp .bar{height:14px;border-radius:3px}
.cmp .b4{background:#9aa5b1}
.cmp .b5{background:#2f6f4f}
.cmp .num{font-size:12px;color:#3e4c59}
.note{background:#f7f9f8;border-left:3px solid #2f6f4f;padding:10px 14px;margin:14px 0;font-size:14px}
.kv{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0}
.kv div{background:#eef4f0;border-radius:6px;padding:8px 14px;font-size:13px}
code{background:#eef4f0;padding:1px 5px;border-radius:3px;font-size:13px}
"""


def _bar_rows(label: str, before: int, after: int, width: int = 300) -> str:
    """一张纯 HTML 的前后对比条（不依赖 JS / 图表库）。"""
    mx = max(before, after, 1)
    wb = int(width * before / mx)
    wa = int(width * after / mx)
    return (
        f'<div class="cmp"><div class="lbl">{label}</div><div class="bars">'
        f'<div class="row"><span class="tag">清洗前</span>'
        f'<div class="bar b4" style="width:{wb}px"></div>'
        f'<span class="num">{before:,}</span></div>'
        f'<div class="row"><span class="tag">清洗后</span>'
        f'<div class="bar b5" style="width:{wa}px"></div>'
        f'<span class="num">{after:,}</span></div>'
        f"</div></div>"
    )


def render_quality_html(report: dict, path: Path | None = None) -> Path:
    """把质量报告渲染成自包含 HTML，返回文件路径。"""
    path = path or (DOCS_DIR / "数据质量报告.html")
    before = report.get("before", {})
    after = report.get("after", {})
    details = report.get("details", {})

    keys = sorted(set(before) | set(after))
    rows = "\n".join(
        f"<tr><td>{k}</td><td>{int(before.get(k, 0)):,}</td>"
        f"<td>{int(after.get(k, 0)):,}</td>"
        f"<td>{int(after.get(k, 0)) - int(before.get(k, 0)):,}</td></tr>"
        for k in keys
    )
    cmps = "\n".join(
        _bar_rows(k, int(before.get(k, 0)), int(after.get(k, 0))) for k in keys
    )

    miss = details.get("missing_rates", {})
    miss_rows = "\n".join(
        f"<tr><td>{k}</td><td>{float(v):.4%}</td></tr>" for k, v in sorted(miss.items())
    ) or '<tr><td colspan="2">清洗后无缺失字段（缺失已转为哨兵值 / 标记列）</td></tr>'

    dedup = int(details.get("dedup_removed", 0))
    total_txn = int(before.get("transactions", 0)) or 1
    kv_items = [
        f"重复行去除 {dedup:,}（占交易 {dedup / total_txn:.3%}）",
        f"缺失会员ID 交易 {int(details.get('missing_member', 0)):,}（归入「非会员」客群，**不删除**）",
        f"退货交易 {int(details.get('returns', 0)):,}",
        f"逻辑冲突 {int(details.get('logic_errors', 0)):,}",
        f"品类别名归一 {int(details.get('alias_fixed', 0)):,}",
        f"极端值阈值 {float(details.get('outlier_threshold', 0)):.2f}（P99.9 截断）",
    ]
    kv_html = "\n".join(f"<div>{x}</div>" for x in kv_items)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>数据质量报告 · RetailInsight</title><style>{_HTML_CSS}</style></head>
<body>
<h1>数据质量报告 · RetailInsight</h1>
<p class="note">本报告由 <code>retailinsight.etl.quality</code> 自动生成，
数值与 <code>docs/数据质量报告.json</code> 同源。</p>

<h2>一、清洗动作概览</h2>
<div class="kv">{kv_html}</div>

<h2>二、清洗前后行数对比</h2>
<table><tr><th>表</th><th>清洗前</th><th>清洗后</th><th>变化</th></tr>
{rows}</table>

<h2>三、行数对比图</h2>
{cmps}

<h2>四、字段缺失率（清洗后）</h2>
<table><tr><th>表.字段</th><th>缺失率</th></tr>
{miss_rows}</table>
<p class="note">说明：缺失在清洗阶段被转为<strong>哨兵值 / 标记列</strong>
（如会员ID 缺失 → 非会员），因此清洗后表内没有 NaN。
这一节展示的是「清洗后仍为空的字段」，而非清洗前的缺失情况。</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
