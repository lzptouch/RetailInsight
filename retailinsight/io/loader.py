"""parquet 读写与 dtype 优化。"""
from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """ID 类降 int32，金额降 float32，低基数字符串转 category（省 60%+ 内存）。"""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if s.dtype == "int64":
            df[col] = pd.to_numeric(s, downcast="integer")
        elif s.dtype == "float64":
            df[col] = pd.to_numeric(s, downcast="float")
        elif s.dtype == object and s.nunique(dropna=False) < max(1000, len(s) // 20):
            df[col] = s.astype("category")
    return df
