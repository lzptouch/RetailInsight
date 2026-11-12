"""开源数据集下载器（计划 §4.1 / R-2）。

原则
----
1. **只下载许可允许的**：需注册或条款受限的数据集只输出下载指引，不做违反条款的抓取。
2. **必须做 sha256 校验**：校验不过立刻删除半包并报错，不让损坏文件进入分析链路。
3. **下载后立刻映射到 canonical schema**：验证「一份 ETL 通吃多源」不是口号。

用法
----
    python scripts/download_data.py --list
    python scripts/download_data.py --source uci_retail            # 下载 + 校验 + 转换
    python scripts/download_data.py --source uci_retail --dry-run  # 只打印将要做什么
    python scripts/download_data.py --source uci_retail --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retailinsight.io.schema import SchemaError, apply_mapping  # noqa: E402

EXTERNAL_DIR = ROOT / "data" / "external"

# 数据源登记：url / sha256 / 许可 / 是否需注册
SOURCES: dict[str, dict] = {
    "uci_retail": {
        "name": "Online Retail II (UCI)",
        "url": ("https://archive.ics.uci.edu/static/public/502/"
                "online+retail+ii.zip"),
        "sha256": None,          # 首次运行后填入实测值（见 --print-sha256）
        "license": "CC BY 4.0",
        "registration_required": False,
        "tables": {"transactions": "transactions.csv", "items": "items.csv"},
        "note": "免注册直下。解压后含 transactions.csv 与 items.csv。",
    },
    "dunnhumby": {
        "name": "The Complete Journey (Dunnhumby)",
        "url": None,
        "sha256": None,
        "license": "仅非商业研究",
        "registration_required": True,
        "tables": {},
        "note": "需在 https://www.dunnhumby.com/completejourney/ 注册并接受条款后下载。",
    },
    "instacart": {
        "name": "Instacart Market Basket",
        "url": None,
        "sha256": None,
        "license": "竞赛条款，仅非商业",
        "registration_required": True,
        "tables": {},
        "note": "需在 Kaggle 接受竞赛规则后下载。",
    },
}

CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> bool:
    """校验 sha256；``expected`` 为 None 时只计算不校验（返回 True）。"""
    actual = sha256_of(path)
    if expected is None:
        print(f"[dl] {path.name} sha256 = {actual}（登记表中无期望值，跳过比对）")
        return True
    if actual != expected:
        raise ValueError(
            f"sha256 校验失败：{path.name}\n  期望 {expected}\n  实际 {actual}")
    print(f"[dl] sha256 校验通过：{actual}")
    return True


def download(url: str, dest: Path) -> Path:
    """流式下载到 ``dest``（先写 ``.part``，成功后再改名，避免半包混入产物）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[dl] 下载 {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
            while chunk := resp.read(CHUNK):
                f.write(chunk)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    print(f"[dl] 已保存 {dest}（{dest.stat().st_size / 1e6:.1f} MB）")
    return dest


def _read_table(path: Path, table: str, source: str) -> pd.DataFrame:
    """读外部表并映射到 canonical；映射失败抛 SchemaError（不静默）。"""
    if table == "transactions":
        df = pd.read_csv(path, encoding="ISO-8859-1", parse_dates=["InvoiceDate"])
        df = df.rename(columns={"InvoiceDate": "InvoiceDate"})
    else:
        df = pd.read_csv(path, encoding="ISO-8859-1")
    return apply_mapping(df, source, table=table)


def fetch(source: str, dest_dir: Path, dry_run: bool = False,
          verify_only: bool = False) -> int:
    """执行一个数据源的下载 → 校验 → 转换。返回退出码。"""
    if source not in SOURCES:
        print(f"未知数据源 {source!r}，可选：{sorted(SOURCES)}")
        return 2
    meta = SOURCES[source]

    if meta["registration_required"]:
        print(f"[dl] {meta['name']} 需要注册/接受条款，**不自动抓取**。")
        print(f"[dl] 获取指引：{meta['note']}")
        print(f"[dl] 许可：{meta['license']}")
        print("[dl] 下载后请放到 " + str(dest_dir) + " 并用 --verify-only 校验。")
        return 0

    if not meta["url"]:
        print(f"[dl] {meta['name']} 未登记直链，请手动下载后放到 {dest_dir}")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / f"{source}.zip"

    if verify_only:
        if not archive.exists():
            print(f"[dl] 文件不存在：{archive}")
            return 1
        verify_sha256(archive, meta["sha256"])
        return 0

    if dry_run:
        print(f"[dl] [dry-run] 将下载 {meta['url']}")
        print(f"[dl] [dry-run] 保存到 {archive}")
        print(f"[dl] [dry-run] 期望 sha256 = {meta['sha256']}")
        return 0

    if not archive.exists():
        download(meta["url"], archive)

    try:
        verify_sha256(archive, meta["sha256"])
    except ValueError as exc:
        archive.unlink(missing_ok=True)   # 校验不过就删，不留坏文件
        print(f"[dl] {exc}")
        print("[dl] 已删除校验失败的文件")
        return 1

    import zipfile
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir / source)

    for table, fname in meta["tables"].items():
        csv_path = dest_dir / source / fname
        if not csv_path.exists():
            print(f"[dl] 解压后未找到 {fname}，跳过 {table}")
            continue
        try:
            out = _read_table(csv_path, table, source)
        except SchemaError as exc:
            print(f"[dl] {source}/{table} 映射失败：{exc}")
            return 1
        out_path = dest_dir / source / f"{table}_canonical.parquet"
        out.to_parquet(out_path, index=False)
        print(f"[dl] {source}/{table} → {out_path.name}（{len(out):,} 行，"
              f"{len(out.columns)} 列）")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="下载开源数据集并校验 sha256")
    parser.add_argument("--source", choices=sorted(SOURCES), help="数据源")
    parser.add_argument("--list", action="store_true", help="列出全部数据源与许可")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的动作")
    parser.add_argument("--verify-only", action="store_true", help="只校验已有文件的 sha256")
    parser.add_argument("--print-sha256", action="store_true",
                        help="打印本地已有文件的 sha256（用于首次登记期望值）")
    args = parser.parse_args()

    if args.list:
        print(f"{'source':<14}{'许可':<16}{'需注册':<8}说明")
        for name, m in SOURCES.items():
            print(f"{name:<14}{m['license']:<16}"
                  f"{'是' if m['registration_required'] else '否':<8}{m['note']}")
        return 0

    if not args.source:
        parser.error("需要 --source（或用 --list 查看全部）")

    if args.print_sha256:
        p = EXTERNAL_DIR / f"{args.source}.zip"
        if not p.exists():
            print(f"文件不存在：{p}")
            return 1
        print(sha256_of(p))
        return 0

    return fetch(args.source, EXTERNAL_DIR, dry_run=args.dry_run,
                 verify_only=args.verify_only)


if __name__ == "__main__":
    raise SystemExit(main())
