# src/experiment_logger.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


LOG_PATH = Path("experiments_log.csv")

HEADER = [
    "timestamp",
    "exp_name",
    "model",
    "seed",
    "n_splits",
    "params",          # JSON with all model-related params (tfidf, lr, transformer, blend, etc.)
    "threshold",
    "oof_f1",
    "oof_roc_auc",
    "oof_accuracy",
    "test_f1",
    "test_roc_auc",
    "test_accuracy",
    "notes",
]


def log_experiment(
    exp_name: str,
    model: str,
    seed: Optional[int],
    n_splits: Optional[int],
    params: Dict[str, Any],
    threshold: Optional[float],
    oof_metrics: Optional[Dict[str, Any]],
    test_metrics: Optional[Dict[str, Any]],
    notes: str = "",
) -> None:
    """
    Unified experiment logger.

    All experiments (TF-IDF, DistilBERT, blends, etc.) log into the same CSV
    with the same header and 'params' JSON field.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_PATH.exists()

    params_str = json.dumps(params or {}, ensure_ascii=False, sort_keys=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_name": exp_name,
        "model": model,
        "seed": seed,
        "n_splits": n_splits,
        "params": params_str,
        "threshold": threshold,
        "oof_f1": None,
        "oof_roc_auc": None,
        "oof_accuracy": None,
        "test_f1": None,
        "test_roc_auc": None,
        "test_accuracy": None,
        "notes": notes,
    }

    if oof_metrics:
        row["oof_f1"] = oof_metrics.get("oof_f1") or oof_metrics.get("f1")
        row["oof_roc_auc"] = oof_metrics.get("oof_roc_auc") or oof_metrics.get("roc_auc")
        row["oof_accuracy"] = oof_metrics.get("oof_accuracy") or oof_metrics.get("accuracy")

    if test_metrics:
        row["test_f1"] = test_metrics.get("f1")
        row["test_roc_auc"] = test_metrics.get("roc_auc")
        row["test_accuracy"] = test_metrics.get("accuracy")

    with LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[log] Experiment appended to {LOG_PATH}")