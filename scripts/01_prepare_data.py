#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import os
import json
import glob
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


from src.config import load_config


def _read_csv_or_json(path: str) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith(".json"):

        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Unsupported file format: {path}")


def _detect_single_file(data_dir: Path) -> str | None:

    for ext in ("csv", "json"):
        candidates = list(data_dir.glob(f"imdb*.{ext}"))
        if candidates:
            return str(candidates[0])

    files = [*data_dir.glob("*.csv"), *data_dir.glob("*.json")]
    if len(files) == 1:
        return str(files[0])
    return None


def _load_from_single_file(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_csv_or_json(path)

    lower_cols = {c.lower(): c for c in df.columns}
    text_col = lower_cols.get("review") or lower_cols.get("text")
    label_col = lower_cols.get("sentiment") or lower_cols.get("label")
    if text_col is None or label_col is None:
        raise ValueError(
            f"Not found cols in {path}. Expected review/text and sentiment/label."
        )
    df = df.rename(columns={text_col: "text", label_col: "label"})
    # строковые метки → 0/1
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"neg": 0, "negative": 0, "pos": 1, "positive": 1})
    df["label"] = df["label"].astype(int)


    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    tr = tr.reset_index(drop=True)
    te = te.reset_index(drop=True)
    return tr, te


def _load_from_pair(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_path = None
    test_path = None
    for ext in ("csv", "json"):
        if (data_dir / f"train.{ext}").exists():
            train_path = str(data_dir / f"train.{ext}")
        if (data_dir / f"test.{ext}").exists():
            test_path = str(data_dir / f"test.{ext}")
    if not train_path or not test_path:
        return None, None  

    tr = _read_csv_or_json(train_path)
    te = _read_csv_or_json(test_path)

    def norm(df: pd.DataFrame) -> pd.DataFrame:
        lower_cols = {c.lower(): c for c in df.columns}
        text_col = lower_cols.get("review") or lower_cols.get("text")
        label_col = lower_cols.get("sentiment") or lower_cols.get("label")
        if text_col is None or label_col is None:
            raise ValueError("In train/test cols not found (review/text, sentiment/label).")
        df = df.rename(columns={text_col: "text", label_col: "label"})
        if df["label"].dtype == object:
            df["label"] = df["label"].str.lower().map({"neg": 0, "negative": 0, "pos": 1, "positive": 1})
        df["label"] = df["label"].astype(int)
        return df.reset_index(drop=True)

    return norm(tr), norm(te)


def _read_txts(root: Path) -> pd.DataFrame:
    rows = []
    for label_name, label_val in (("pos", 1), ("neg", 0)):
        for fp in glob.glob(str(root / label_name / "*.txt")):
            try:
                text = Path(fp).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = Path(fp).read_text(encoding="latin-1")
            rows.append({"text": text, "label": label_val})
    if not rows:
        return pd.DataFrame(columns=["text", "label"])
    return pd.DataFrame(rows)


def _load_from_folders(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        return None, None
    tr = _read_txts(train_dir)
    te = _read_txts(test_dir)
    if tr.empty or te.empty:
        raise ValueError("Folders train/test found, but empty or no .txt в pos/neg.")
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def build_folds(train_df: pd.DataFrame, n_splits: int, seed: int) -> List[dict]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, va_idx in skf.split(train_df["text"], train_df["label"]):
        folds.append({"train_idx": tr_idx.tolist(), "valid_idx": va_idx.tolist()})
    return folds


def main():
    parser = argparse.ArgumentParser(description="Prepare local IMDb-like dataset")
    parser.add_argument("--data_dir", default="data", help="Path to original data")
    parser.add_argument("--out_dir", default="data", help="Where to save train.csv/test.csv/cv_folds.json")
    args, _ = parser.parse_known_args()

    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    n_splits = int(cfg.get("data", {}).get("n_splits", 5))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) пара train/test файлов?
    train_df, test_df = _load_from_pair(data_dir)
    if train_df is not None:
        print("✔ Found files train/test (csv/json).")
    else:
        # 2) директории pos/neg?
        train_df, test_df = _load_from_folders(data_dir)
        if train_df is not None:
            print("✔ Found folders train/test with pos/neg .txt files.")
        else:
            # 3) один общий файл?
            single = _detect_single_file(data_dir)
            if single:
                print(f"✔ Single file found: {single} — splitting in train/test.")
                train_df, test_df = _load_from_single_file(single)
            else:
                raise SystemExit()



    train_df["text"] = train_df["text"].astype(str).str.strip()
    test_df["text"] = test_df["text"].astype(str).str.strip()


    train_out = out_dir / "train.csv"
    test_out = out_dir / "test.csv"
    folds_out = out_dir / "cv_folds.json"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    folds = build_folds(train_df, n_splits=n_splits, seed=seed)
    with open(folds_out, "w", encoding="utf-8") as f:
        json.dump(folds, f)

    print(f"✅ Saved:\n- {train_out}\n- {test_out}\n- {folds_out}")
    print(f"Sizes: train={len(train_df)}, test={len(test_df)}, folds={len(folds)} (n_splits={n_splits})")


if __name__ == "__main__":
    main()