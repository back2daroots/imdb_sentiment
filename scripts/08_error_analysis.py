#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
08_error_analysis.py

Error analysis for TF-IDF+LR vs DistilBERT vs Blend:
- recompute test probabilities for TF-IDF and DistilBERT
- recompute best alpha for blending on OOF (as in 07_blend_models.py)
- create a CSV with per-sample predictions and categories:
    * both_correct
    * both_wrong
    * tfidf_wrong_trf_correct
    * tfidf_correct_trf_wrong
- also includes blended predictions
- saves:
    reports/error_analysis_all.csv
    reports/error_analysis_tfidf_only_wrong.csv
    reports/error_analysis_trf_only_wrong.csv
    reports/error_analysis_both_wrong.csv
"""

import sys
from pathlib import Path
from typing import Tuple

import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# make src importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.config import load_config  # noqa: E402

import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def ensure_dirs():
    Path("reports").mkdir(exist_ok=True)


def load_oof_predictions() -> Tuple[np.ndarray, np.ndarray]:
    oof_tfidf_path = Path("models/oof_tfidf_lr.npy")
    oof_trf_path = Path("models/oof_distilbert.npy")

    if not oof_tfidf_path.exists():
        raise SystemExit(f"Missing {oof_tfidf_path}, run 02_train_tfidf_lr.py first.")
    if not oof_trf_path.exists():
        raise SystemExit(f"Missing {oof_trf_path}, run 05_train_distilbert.py first.")

    oof_tfidf = np.load(oof_tfidf_path)
    oof_trf = np.load(oof_trf_path)

    if oof_tfidf.shape != oof_trf.shape:
        raise SystemExit(f"Shape mismatch: oof_tfidf {oof_tfidf.shape} vs oof_distilbert {oof_trf.shape}")

    return oof_tfidf, oof_trf


def search_best_alpha(y_true, oof_tfidf, oof_trf, threshold=0.5, alphas=None):
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, ..., 1.0

    best = {
        "alpha": None,
        "f1": -1.0,
        "roc_auc": None,
        "accuracy": None,
    }

    for a in alphas:
        p_blend = (1 - a) * oof_tfidf + a * oof_trf
        preds = (p_blend >= threshold).astype(int)
        f1 = f1_score(y_true, preds)
        roc_auc = roc_auc_score(y_true, p_blend)
        acc = accuracy_score(y_true, preds)

        if f1 > best["f1"]:
            best = {
                "alpha": a,
                "f1": f1,
                "roc_auc": roc_auc,
                "accuracy": acc,
            }

    return best


def resolve_transformer_model_dir(fold_dir: Path) -> Path:
    """
    Try to find a valid HF checkpoint:
    - if fold_dir has config.json -> use it
    - else pick the last checkpoint-* subdir
    """
    cfg_path = fold_dir / "config.json"
    if cfg_path.exists():
        return fold_dir

    checkpoints = sorted(fold_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise SystemExit(f"No config.json or checkpoint-* dirs found in {fold_dir}")
    return checkpoints[-1]


def compute_test_probas_tfidf(test_df: pd.DataFrame) -> np.ndarray:
    paths = sorted(Path("models").glob("lr_fold*.joblib"))
    if not paths:
        raise SystemExit("No lr_fold*.joblib found. Run 02_train_tfidf_lr.py first.")

    texts = test_df["text"].astype(str).tolist()
    probas = np.zeros(len(test_df), dtype=float)

    for p in paths:
        model = joblib.load(p)
        probas += model.predict_proba(texts)[:, 1]

    probas /= len(paths)
    return probas


def compute_test_probas_distilbert(test_df: pd.DataFrame, cfg: dict) -> np.ndarray:
    trf_cfg = cfg.get("model", {}).get("transformer", {})
    model_name = trf_cfg.get("model_name", "distilbert-base-uncased")
    max_length = int(trf_cfg.get("max_length", 256))
    batch_size = int(trf_cfg.get("eval_batch_size", 32))

    fold_dirs = sorted(Path("models").glob("transformer_fold*"))
    if not fold_dirs:
        raise SystemExit("No transformer_fold* directories found. Run 05_train_distilbert.py first.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = test_df["text"].astype(str).tolist()
    all_probas = np.zeros((len(fold_dirs), len(test_df)), dtype=float)

    for fi, fold_dir in enumerate(fold_dirs):
        model_dir = resolve_transformer_model_dir(fold_dir)
        print(f"[error_analysis] Loading transformer fold from {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        probs = []

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
    return probas_mean


def main():
    cfg = load_config()
    ensure_dirs()

    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))

    # === 1. Load train + OOF and find best alpha ===
    train_df = pd.read_csv("data/train.csv")
    y_train = train_df["label"].astype(int).to_numpy()

    oof_tfidf, oof_trf = load_oof_predictions()
    best_oof = search_best_alpha(y_train, oof_tfidf, oof_trf, threshold=threshold)
    alpha = best_oof["alpha"]

    print("== Error analysis: best OOF blend ==")
    print(f"alpha:    {alpha:.3f}")
    print(f"F1:       {best_oof['f1']:.4f}")
    print(f"ROC-AUC:  {best_oof['roc_auc']:.4f}")
    print(f"Accuracy: {best_oof['accuracy']:.4f}")

    # === 2. Compute test probabilities for both models ===
    test_df = pd.read_csv("data/test.csv")
    y_test = test_df["label"].astype(int).to_numpy()
    texts = test_df["text"].astype(str).tolist()

    print("\n[error_analysis] Computing TF-IDF test probabilities...")
    probas_tfidf = compute_test_probas_tfidf(test_df)

    print("[error_analysis] Computing DistilBERT test probabilities...")
    probas_trf = compute_test_probas_distilbert(test_df, cfg)

    # blended
    probas_blend = (1 - alpha) * probas_tfidf + alpha * probas_trf

    preds_tfidf = (probas_tfidf >= threshold).astype(int)
    preds_trf = (probas_trf >= threshold).astype(int)
    preds_blend = (probas_blend >= threshold).astype(int)

    # quick sanity-check metrics on test (optional print)
    print("\n== Quick test sanity-check ==")
    for name, probas, preds in [
        ("TF-IDF", probas_tfidf, preds_tfidf),
        ("DistilBERT", probas_trf, preds_trf),
        ("Blend", probas_blend, preds_blend),
    ]:
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probas)
        acc = accuracy_score(y_test, preds)
        print(f"{name:10s}: F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, Acc={acc:.4f}")

    # === 3. Build DataFrame with categories ===
    df = pd.DataFrame({
        "text": texts,
        "label": y_test,
        "proba_tfidf": probas_tfidf,
        "pred_tfidf": preds_tfidf,
        "proba_trf": probas_trf,
        "pred_trf": preds_trf,
        "proba_blend": probas_blend,
        "pred_blend": preds_blend,
    })

    cases = []
    for i in range(len(df)):
        y_true = df.loc[i, "label"]
        pt = df.loc[i, "pred_tfidf"]
        pr = df.loc[i, "pred_trf"]

        if pt == y_true and pr == y_true:
            case = "both_correct"
        elif pt != y_true and pr != y_true:
            case = "both_wrong"
        elif pt != y_true and pr == y_true:
            case = "tfidf_wrong_trf_correct"
        elif pt == y_true and pr != y_true:
            case = "tfidf_correct_trf_wrong"
        else:
            case = "other" 

        cases.append(case)

    df["case"] = cases

    reports_dir = Path("reports")

    all_path = reports_dir / "error_analysis_all.csv"
    tfidf_only_wrong_path = reports_dir / "error_analysis_tfidf_only_wrong.csv"
    trf_only_wrong_path = reports_dir / "error_analysis_trf_only_wrong.csv"
    both_wrong_path = reports_dir / "error_analysis_both_wrong.csv"

    df.to_csv(all_path, index=False)

    df_tfidf_wrong = df[df["case"] == "tfidf_wrong_trf_correct"]
    df_trf_wrong = df[df["case"] == "tfidf_correct_trf_wrong"]
    df_both_wrong = df[df["case"] == "both_wrong"]

    df_tfidf_wrong.to_csv(tfidf_only_wrong_path, index=False)
    df_trf_wrong.to_csv(trf_only_wrong_path, index=False)
    df_both_wrong.to_csv(both_wrong_path, index=False)

    # === 4. Print brief summary ===
    print("\n== Error cases summary ==")
    print(df["case"].value_counts())

    print("\nSaved:")
    print(f"- {all_path}")
    print(f"- {tfidf_only_wrong_path}   # where DistilBERT fixes TF-IDF mistakes")
    print(f"- {trf_only_wrong_path}   # where TF-IDF is better than DistilBERT")
    print(f"- {both_wrong_path}      # hard cases for both models")


if __name__ == "__main__":
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score  # noqa: E402
    main()