"""下载器与 schema 映射的端到端测试。

用 ``file://`` URL 回环，完整跑通「下载 → sha256 校验 → 解压 → 映射 canonical」，
不依赖外网。
"""
import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import scripts.download_data as dl
from retailinsight.io.schema import SchemaError, apply_mapping


@pytest.fixture()
def uci_zip(tmp_path: Path) -> Path:
    """构造一个结构与 UCI Online Retail II 一致的最小 zip 包。"""
    d = tmp_path / "src"
    d.mkdir()
    txn = pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "536366"],
        "StockCode": ["85123A", "71053", "22752"],
        "Quantity": [6, 6, 2],
        "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-01 08:26:00", "2010-12-01 08:34:00"],
        "Price": [2.55, 2.55, 3.39],
        "CustomerID": [17850.0, 17850.0, 17850.0],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER", "SET 7 BABUSHKA NURSERY", "RED WOOLLY HOTTIE WHITE HEART."],
    })
    items = pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "536366"],
        "StockCode": ["85123A", "71053", "22752"],
        "Quantity": [6, 6, 2],
        "UnitPrice": [2.55, 2.55, 3.39],
    })
    txn.to_csv(d / "transactions.csv", index=False)
    items.to_csv(d / "items.csv", index=False)

    zpath = tmp_path / "uci_retail.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in ("transactions.csv", "items.csv"):
            zf.write(d / f, arcname=f)
    return zpath


# ---------------------------------------------------------------- 基础工具

def test_sha256_of_matches_hashlib(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello retailinsight")
    assert dl.sha256_of(f) == hashlib.sha256(b"hello retailinsight").hexdigest()


def test_verify_sha256_passes(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 100)
    assert dl.verify_sha256(f, dl.sha256_of(f)) is True


def test_verify_sha256_raises_on_mismatch(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="sha256 校验失败"):
        dl.verify_sha256(f, "0" * 64)


def test_verify_sha256_without_expected_only_computes(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x")
    assert dl.verify_sha256(f, None) is True


# ---------------------------------------------------------------- 下载流程

def test_fetch_downloads_verifies_and_converts(uci_zip: Path, tmp_path: Path, monkeypatch):
    """完整链路：file:// 下载 → sha256 校验 → 解压 → 映射 canonical。"""
    expected = dl.sha256_of(uci_zip)
    monkeypatch.setitem(dl.SOURCES["uci_retail"], "sha256", expected)
    monkeypatch.setitem(dl.SOURCES["uci_retail"], "url", uci_zip.as_uri())

    dest = tmp_path / "external"
    code = dl.fetch("uci_retail", dest)
    assert code == 0

    txn_out = dest / "uci_retail" / "transactions_canonical.parquet"
    items_out = dest / "uci_retail" / "items_canonical.parquet"
    assert txn_out.exists() and items_out.exists()

    txn = pd.read_parquet(txn_out)
    assert "member_id" in txn.columns
    assert "txn_id" in txn.columns
    assert "txn_ts" in txn.columns
    assert "CustomerID" not in txn.columns
    assert str(txn["txn_ts"].dtype).startswith("datetime64")

    items = pd.read_parquet(items_out)
    assert set(["txn_id", "sku", "qty", "unit_price"]).issubset(items.columns)


def test_fetch_deletes_file_on_sha256_mismatch(uci_zip: Path, tmp_path: Path, monkeypatch):
    """校验不过必须删文件，不留坏包混入分析链路。"""
    monkeypatch.setitem(dl.SOURCES["uci_retail"], "sha256", "f" * 64)
    monkeypatch.setitem(dl.SOURCES["uci_retail"], "url", uci_zip.as_uri())

    dest = tmp_path / "external"
    code = dl.fetch("uci_retail", dest)
    assert code == 1
    assert not (dest / "uci_retail.zip").exists()


def test_fetch_registration_required_source_skips_download(tmp_path: Path):
    code = dl.fetch("dunnhumby", tmp_path)
    assert code == 0
    assert not (tmp_path / "dunnhumby.zip").exists()


def test_fetch_unknown_source_returns_error_code(tmp_path: Path):
    assert dl.fetch("nope", tmp_path) == 2


# ---------------------------------------------------------------- 映射契约

def test_read_table_maps_uci_transactions(tmp_path: Path):
    csv = tmp_path / "transactions.csv"
    pd.DataFrame({
        "InvoiceNo": ["1"], "StockCode": ["A"], "Quantity": [1],
        "InvoiceDate": ["2011-01-05 09:00:00"], "Price": [1.0], "CustomerID": [12345.0],
    }).to_csv(csv, index=False)

    out = dl._read_table(csv, "transactions", "uci_retail")
    assert {"txn_id", "member_id", "txn_ts", "sku", "qty", "unit_price"}.issubset(out.columns)
    assert out["txn_ts"].iloc[0].year == 2011
    assert out["member_id"].iloc[0] == 12345


def test_read_table_raises_schema_error_on_bad_source(tmp_path: Path):
    csv = tmp_path / "items.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(csv, index=False)
    with pytest.raises(SchemaError):
        dl._read_table(csv, "items", "uci_retail")


def test_synthetic_source_is_identity_mapping():
    df = pd.DataFrame({"member_id": [1], "txn_id": [2], "sku": [3]})
    out = apply_mapping(df, "synthetic")
    assert list(out.columns) == list(df.columns)
