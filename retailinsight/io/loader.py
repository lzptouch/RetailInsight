"""parquet 读写、分区读写与 dtype 优化。"""
from pathlib import Path

import numpy as np
import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def write_partitioned(df: pd.DataFrame, dir_path: Path, partition_col: str) -> Path:
    """按 ``partition_col`` 分区写 parquet（hive 风格 ``dir/col=value/*.parquet``）。

    分区的意义：**窗口期分析只加载所需分区**，不必把全量明细读进内存。
    写入前会清空同名分区目录，避免残留旧分区造成数据不一致。
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for stale in dir_path.iterdir():
        if stale.is_dir() and stale.name.startswith(f"{partition_col}="):
            for f in stale.rglob("*"):
                if f.is_file():
                    f.unlink()
            stale.rmdir()
    # 逐分区写：一次只把一个分区转成 arrow，避免整表转换时的内存峰值
    # （分区值编码在目录名里，文件内不再重复存该列）
    for value, sub in df.groupby(partition_col, observed=True, sort=True):
        target = dir_path / f"{partition_col}={value}"
        target.mkdir(parents=True, exist_ok=True)
        sub.drop(columns=[partition_col]).to_parquet(target / "part-0.parquet", index=False)
    return dir_path


def append_partitioned(df: pd.DataFrame, dir_path: Path, partition_col: str,
                       part_name: str) -> Path:
    """向已有分区目录**追加**一个 part 文件（不清空旧分区）。

    `write_partitioned` 是「整体重写」语义（先清空再写 part-0），适合一次性生成；
    但大规模生成必须**分块流式落盘**，否则千万级明细会把内存打爆。
    分块写入时每个块各写一个 ``part-{i}.parquet``，读侧
    （`pd.read_parquet(目录)`）会自动把同分区的多个 part 合并。

    ``part_name`` 同名文件会先删除，保证同一 chunk 重复执行不会产生重复数据。
    """
    dir_path = Path(dir_path)
    for value, sub in df.groupby(partition_col, observed=True, sort=True):
        target = dir_path / f"{partition_col}={value}"
        target.mkdir(parents=True, exist_ok=True)
        f = target / f"{part_name}.parquet"
        if f.exists():
            f.unlink()
        sub.drop(columns=[partition_col]).to_parquet(f, index=False)
    return dir_path


def concat_parquet_parts(parts: list[Path], target: Path, part_col: str = "txn_month") -> Path:
    """把多个 parquet part **流式**合并成一个文件（每次只驻留一个 part 在内存）。

    为什么不用 ``pd.concat``：大规模下所有 part 同时驻留内存正是要避免的事。
    这里用 `pq.ParquetWriter` 顺序 write_table，峰值 ≈ 单个 part。
    """
    import pyarrow.parquet as pq

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        return target
    writer = None
    try:
        for p in parts:
            table = pq.read_table(p)
            if part_col in table.column_names:
                table = table.drop([part_col])
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return target


def read_parquet(path: Path, columns: list[str] | None = None,
                 filters: list | None = None) -> pd.DataFrame:
    """读 parquet；``path`` 是目录时按分区数据集读（支持 ``filters`` 下推）。"""
    path = Path(path)
    if path.is_dir():
        return pd.read_parquet(path, columns=columns, filters=filters)
    return pd.read_parquet(path, columns=columns)


def read_partitioned(df: pd.DataFrame, partition_col: str, months: list[str] | None) -> pd.DataFrame:
    """在内存里按月份过滤分区列（``months`` 为 None 表示不过滤）。"""
    if not months or partition_col not in df.columns:
        return df
    return df[df[partition_col].astype(str).isin([str(m) for m in months])]


def _is_string_dtype(s: pd.Series) -> bool:
    """兼容 pandas 3.x 的 ``str`` dtype 与旧版的 ``object`` dtype。"""
    return s.dtype == object or str(s.dtype) in ("str", "string", "string[python]")


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """ID 类降位，金额降 float32，低基数字符串转 category（省 60%+ 内存）。

    注意：pandas 3.x 起字符串列默认 dtype 是 ``str`` 而不是 ``object``，
    只判 ``object`` 会让分类转换**静默失效**（内存优化形同虚设）。
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if s.dtype == "int64":
            df[col] = pd.to_numeric(s, downcast="integer")
        elif s.dtype == "float64":
            df[col] = pd.to_numeric(s, downcast="float")
        elif _is_string_dtype(s) and s.nunique(dropna=False) < max(1000, len(s) // 20):
            df[col] = s.astype("category")
    return df


def derive_txn_month(items: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    """给明细表补 ``txn_month`` 分区列（YYYY-MM），取自交易表的 ``txn_ts``。

    明细表本身不含时间戳，必须通过 ``txn_id`` 关联交易表才能分区。

    **内存注意**：用 ``Series.map`` 会把 3500 万行展开成 3500 万个 Python 字符串
    对象（scale 5 时额外占用 2GB+，实测 ETL 在该规模下失败）。这里改用
    **int16 编码 + Categorical**，内存约为对象列的 1/20。
    """
    if "txn_month" in items.columns:
        return items

    txn_idx = txn.drop_duplicates("txn_id").set_index("txn_id")
    months = txn_idx["txn_ts"].dt.strftime("%Y-%m")
    cats = sorted(set(months.tolist()) | {"unknown"})
    month_code = {m: i for i, m in enumerate(cats)}
    unknown_code = month_code["unknown"]

    ids = items["txn_id"].to_numpy()
    max_id = int(txn_idx.index.max()) if len(txn_idx) else 0
    lut = np.full(max_id + 1, unknown_code, dtype=np.int16)
    pos = txn_idx.index.to_numpy()
    lut[pos] = np.array([month_code[m] for m in months.tolist()], dtype=np.int16)
    codes = np.where(ids <= max_id, lut[np.clip(ids, 0, max_id)], unknown_code)

    out = items.copy()
    out["txn_month"] = pd.Categorical.from_codes(codes.astype("int16"), categories=cats)
    return out


def read_items(path: Path, months: list[str] | None = None,
               columns: list[str] | None = None) -> pd.DataFrame:
    """读明细表（分区优先）。

    - ``path`` 是分区目录 → 按需只读指定月份分区
    - ``path`` 是单个 parquet → 退化为整表读（兼容旧产物）
    """
    path = Path(path)
    if path.is_dir():
        filters = [("txn_month", "in", [str(m) for m in months])] if months else None
        return read_parquet(path, columns=columns, filters=filters)
    return read_parquet(path, columns=columns)
