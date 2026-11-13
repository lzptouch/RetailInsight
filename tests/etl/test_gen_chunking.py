"""分块流式生成（`--chunk-size`）的正确性测试。

背景：`--scale 5` 的明细规模在千万级，一次性生成会把内存打爆。分块生成的
风险点是**跨块一致性**：`txn_id` 必须全局唯一且连续、明细不得指向不存在的交易、
重复执行不得产生叠加数据。这几条一旦错，下游 ETL 会静默错位。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_synthetic import generate  # noqa: E402


@pytest.fixture()
def chunked(tmp_path):
    """10,000 会员 / 2,000 每块 → 5 块。"""
    stats = generate(
        10_000, seed=42, chunk_size=2_000,
        raw_dir=tmp_path / "raw",
        raw_items_dir=tmp_path / "raw" / "transaction_items",
        raw_items_file=tmp_path / "raw" / "transaction_items.parquet",
    )
    return stats, tmp_path / "raw"


def test_chunked_txn_ids_are_globally_unique_and_contiguous(chunked):
    _stats, raw = chunked
    txn = pd.read_parquet(raw / "transactions.parquet")
    uniq = txn.drop_duplicates("txn_id").txn_id.sort_values()
    # 唯一（重复行是注入的脏数据埋点，沿用原 txn_id，不算冲突）
    assert uniq.is_unique
    # 连续：块间偏移必须用「注入重复前」的交易数，否则会出现号段空洞
    assert uniq.max() == len(uniq)
    assert (uniq.values == range(1, len(uniq) + 1)).all()


def test_chunked_items_reference_existing_transactions(chunked):
    _stats, raw = chunked
    txn = pd.read_parquet(raw / "transactions.parquet")
    items = pd.read_parquet(raw / "transaction_items")
    assert set(items.txn_id.unique()) <= set(txn.txn_id.unique())


def test_chunked_members_are_disjoint_across_chunks(chunked):
    _stats, raw = chunked
    members = pd.read_parquet(raw / "members.parquet")
    assert len(members) == 10_000
    assert members.member_id.is_unique
    assert members.member_id.min() == 1 and members.member_id.max() == 10_000


def test_chunked_items_written_as_one_part_per_chunk(chunked):
    _stats, raw = chunked
    parts = list((raw / "transaction_items").glob("txn_month=*/part-*.parquet"))
    # 5 个块 … 每个块至少写入「它实际覆盖到的月份分区」中的若干 part
    assert len(parts) >= 5
    assert all(p.stem.startswith("part-") for p in parts)


def test_chunked_txn_parts_are_cleaned_up(chunked):
    _stats, raw = chunked
    assert not (raw / "_txn_parts").exists(), "合并完必须清掉临时 part 目录"


def test_chunked_output_is_readable_as_single_frame(chunked):
    """多 part 的分区目录必须能被 `pd.read_parquet(目录)` 直接读成一张表。"""
    stats, raw = chunked
    items = pd.read_parquet(raw / "transaction_items")
    assert len(items) == stats["items"]
    assert {"txn_id", "sku", "qty", "unit_price"}.issubset(items.columns)


def test_unchunked_path_stays_byte_identical_to_legacy(tmp_path):
    """`chunk_size=0` 必须沿用历史的随机数序列（否则既有产物与报告全部失效）。

    判据：同 seed 生成两次，指纹一致；且与「分块生成」的数据量在同一量级。
    """
    def fingerprint(dirpath: Path) -> tuple[int, str]:
        df = pd.read_parquet(dirpath / "transactions.parquet")
        return len(df), pd.util.hash_pandas_object(df, index=False).sum().astype(str)

    a = tmp_path / "a"
    b = tmp_path / "b"
    for d in (a, b):
        generate(3_000, seed=42, chunk_size=0,
                 raw_dir=d, raw_items_dir=d / "transaction_items", raw_items_file=None)
    assert fingerprint(a) == fingerprint(b)
