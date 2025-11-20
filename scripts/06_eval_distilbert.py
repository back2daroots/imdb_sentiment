import os
import sys
import glob
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.config import load_config 

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def ensure_dirs():
    Path("reports").mkdir(exist_ok=True)

def plot_confusion(cm: np.ndarray, out_path: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (test, DistilBERT)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["neg", "pos"])
    plt.yticks(tick_marks, ["neg", "pos"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def log_experiment(test_metrics: dict, cfg: dict, notes: str = ""):
    exp_path = Path("experiments_log.csv")

    cv_metrics_path = Path("reports/cv_metrics_distilbert.json")
    oof_f1 = oof_roc_auc = oof_accuracy = None
    trf_params = {}
    seed = cfg.get("seed", None)
    n_splits = cfg.get("data", {}).get("n_splits", None)
    threshold = cfg.get("eval", {}).get("threshold", None)

    if cv_metrics_path.exists():
        with cv_metrics_path.open("r", encoding="utf-8") as f:
            cv = json.load(f)
        oof_f1 = cv.get("oof_f1")
        oof_roc_auc = cv.get("oof_roc_auc")
        oof_accuracy = cv.get("oof_accuracy")
        trf_params = cv.get("transformer", cfg.get("model", {}).get("transformer", {}))
        if seed is None:
            seed = cv.get("seed", None)

    trf_str = json.dumps(trf_params, ensure_ascii=False, sort_keys=True)

    exp_name = cfg.get("eval", {}).get("exp_name", "distilbert_baseline")
    model_name = trf_params.get("model_name", "distilbert-base-uncased")

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_name": exp_name,
        "model": model_name,
        "seed": seed,
        "n_splits": n_splits,
        "params": trf_str,
        "threshold": threshold,
        "oof_f1": oof_f1,
        "oof_roc_auc": oof_roc_auc,
        "oof_accuracy": oof_accuracy,
        "test_f1": test_metrics.get("f1"),
        "test_roc_auc": test_metrics.get("roc_auc"),
        "test_accuracy": test_metrics.get("accuracy"),
        "notes": notes,
    }

    header = list(row.keys())
    file_exists = exp_path.exists()

    with exp_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[log] DistilBERT experiment appended to {exp_path}")

def main():
    cfg = load_config()
    trf_cfg = cfg.get("model", {}).get("transformer", {})
    model_name = trf_cfg.get("model_name", "distilbert-base-uncased")
    max_length = int(trf_cfg.get("max_length", 256))
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))

    ensure_dirs()

    test = pd.read_csv("data/test.csv")
    texts = test["text"].astype(str).tolist()
    y_true = test["label"].astype(int).to_numpy()


    fold_dirs = sorted(Path("models").glob("transformer_fold*"))
    if not fold_dirs:
        raise SystemExit("No transformer_fold* directories found in models/. Run 05_train_distilbert.py first.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_probas = np.zeros((len(fold_dirs), len(test)), dtype=float)

    for fi, fold_dir in enumerate(fold_dirs):
        print(f"Loading fold model from {fold_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(fold_dir)
        model.to(device)
        model.eval()

        probs = []

        batch_size = int(trf_cfg.get("eval_batch_size", 32))
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits
                probs_batch = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.append(probs_batch)

        probs = np.concatenate(probs, axis=0)
        all_probas[fi] = probs

    probas_mean = all_probas.mean(axis=0)
    preds = (probas_mean >= threshold).astype(int)

    metrics = {
        "f1": float(f1_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probas_mean)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "threshold": threshold,
        "n_models": len(fold_dirs),
    }

    with Path("reports/test_metrics_distilbert.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    cls_report = classification_report(y_true, preds, output_dict=True)
    with Path("reports/test_classification_report_distilbert.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(cls_report, f, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_true, preds)
    plot_confusion(cm, "reports/confusion_matrix_distilbert.png")

    print("== DistilBERT Test metrics ==")
    print(f"F1:       {metrics['f1']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        "Saved: reports/test_metrics_distilbert.json, "
        "reports/test_classification_report_distilbert.json, "
        "reports/confusion_matrix_distilbert.png"
    )

    log_experiment(
        test_metrics=metrics,
        cfg=cfg,
        notes="DistilBERT baseline with K-Fold ensemble",
    )


if __name__ == "__main__":
    main()