#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
09_conditional_blend.py

Conditional blending TF-IDF and DistilBERT based on text length.

Rule:
  if len_words(text) <= N:
      p = (1 - alpha_short) * p_tfidf + alpha_short * p_trf
  else:
      p = (1 - alpha_long)  * p_tfidf + alpha_long  * p_trf

We search (N, alpha_short, alpha_long) on OOF to maximize F1.
Then we recompute TF-IDF and DistilBERT test probabilities and evaluate on test.

Outputs:
  reports/conditional_blend_metrics.json
  reports/confusion_matrix_conditional_blend.png
  logs into experiments_log.csv
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

# make src importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.config import load_config  # noqa: E402
from src.experiment_logger import log_experiment  # noqa: E402

import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def ensure_dirs():
    Path("reports").mkdir(exist_ok=True)


def len_words(text: str) -> int:
    return len(str(text).split())


def load_oof_predictions() -> Tuple[np.ndarray, np.ndarray]:
    p1 = Path("models/oof_tfidf_lr.npy")
    p2 = Path("models/oof_distilbert.npy")
    if not p1.exists():
        raise SystemExit(f"Missing {p1} (run 02_train_tfidf_lr.py)")
    if not p2.exists():
        raise SystemExit(f"Missing {p2} (run 05_train_distilbert.py)")
    oof_tfidf = np.load(p1)
    oof_trf = np.load(p2)
    if oof_tfidf.shape != oof_trf.shape:
        raise SystemExit(f"OOF shape mismatch: {oof_tfidf.shape} vs {oof_trf.shape}")
    return oof_tfidf, oof_trf


def apply_conditional_blend(
    prob_tfidf: np.ndarray,
    prob_trf: np.ndarray,
    lengths: np.ndarray,
    N: int,
    alpha_short: float,
    alpha_long: float,
) -> np.ndarray:
    p = np.empty_like(prob_tfidf, dtype=float)
    mask_short = lengths <= N
    p[mask_short] = (1 - alpha_short) * prob_tfidf[mask_short] + alpha_short * prob_trf[mask_short]
    p[~mask_short] = (1 - alpha_long) * prob_tfidf[~mask_short] + alpha_long * prob_trf[~mask_short]
    return p


def score_probs(y: np.ndarray, probas: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probas >= threshold).astype(int)
    return {
        "f1": float(f1_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, probas)),
        "accuracy": float(accuracy_score(y, preds)),
    }


def search_best_params(
    y: np.ndarray,
    oof_tfidf: np.ndarray,
    oof_trf: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    N_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> Dict[str, Any]:
    best = {"f1": -1.0}

    for N in N_grid:
        for a_s in alpha_grid:
            for a_l in alpha_grid:
                prob = apply_conditional_blend(oof_tfidf, oof_trf, lengths, int(N), float(a_s), float(a_l))
                preds = (prob >= threshold).astype(int)
                f1 = f1_score(y, preds)
                if f1 > best["f1"]:
                    m = score_probs(y, prob, threshold)
                    best = {
                        "N": int(N),
                        "alpha_short": float(a_s),
                        "alpha_long": float(a_l),
                        "f1": m["f1"],
                        "roc_auc": m["roc_auc"],
                        "accuracy": m["accuracy"],
                    }

    return best


def resolve_transformer_model_dir(fold_dir: Path) -> Path:
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
        print(f"[conditional_blend] Loading transformer fold from {model_dir}")
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
                logits = model(**enc).logits
                probs_batch = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.append(probs_batch)
        all_probas[fi] = np.concatenate(probs, axis=0)

    return all_probas.mean(axis=0)


def plot_confusion(cm: np.ndarray, out_path: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (test, conditional blend)")
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


def main():
    cfg = load_config()
    ensure_dirs()

    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))

    # === Train side: search best (N, alpha_short, alpha_long) on OOF ===
    train_df = pd.read_csv("data/train.csv")
    y_train = train_df["label"].astype(int).to_numpy()
    lengths_train = train_df["text"].astype(str).apply(len_words).to_numpy()

    oof_tfidf, oof_trf = load_oof_predictions()

    # reasonable grids (fast + effective)
    # N in words, picked around typical short/long separation
    N_grid = np.array([20, 40, 60, 80, 120, 160, 200])
    alpha_grid = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0

    best = search_best_params(
        y=y_train,
        oof_tfidf=oof_tfidf,
        oof_trf=oof_trf,
        lengths=lengths_train,
        threshold=threshold,
        N_grid=N_grid,
        alpha_grid=alpha_grid,
    )

    print("\n== Best conditional blend on OOF ==")
    print(f"N (words):     {best['N']}")
    print(f"alpha_short:   {best['alpha_short']:.2f}")
    print(f"alpha_long:    {best['alpha_long']:.2f}")
    print(f"OOF F1:        {best['f1']:.4f}")
    print(f"OOF ROC-AUC:   {best['roc_auc']:.4f}")
    print(f"OOF Accuracy:  {best['accuracy']:.4f}")

    # === Test side: recompute model probabilities and apply rule ===
    test_df = pd.read_csv("data/test.csv")
    y_test = test_df["label"].astype(int).to_numpy()
    lengths_test = test_df["text"].astype(str).apply(len_words).to_numpy()

    print("\n[conditional_blend] Computing TF-IDF test probabilities...")
    prob_tfidf_test = compute_test_probas_tfidf(test_df)

    print("[conditional_blend] Computing DistilBERT test probabilities...")
    prob_trf_test = compute_test_probas_distilbert(test_df, cfg)

    prob_cond = apply_conditional_blend(
        prob_tfidf=prob_tfidf_test,
        prob_trf=prob_trf_test,
        lengths=lengths_test,
        N=best["N"],
        alpha_short=best["alpha_short"],
        alpha_long=best["alpha_long"],
    )

    test_metrics = score_probs(y_test, prob_cond, threshold)

    preds_cond = (prob_cond >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds_cond)
    plot_confusion(cm, "reports/confusion_matrix_conditional_blend.png")

    out = {
        "best_oof": best,
        "test_metrics": test_metrics,
        "threshold": threshold,
    }
    with Path("reports/conditional_blend_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n== Conditional Blend Test metrics ==")
    print(f"F1:       {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print("Saved: reports/conditional_blend_metrics.json, reports/confusion_matrix_conditional_blend.png")

    # unified log
    seed = cfg.get("seed", None)
    n_splits = cfg.get("data", {}).get("n_splits", None)
    exp_name = cfg.get("eval", {}).get("exp_name", "conditional_blend_len_words")

    params = {
        "rule": "if len_words<=N then alpha_short else alpha_long",
        "N_words": best["N"],
        "alpha_short": best["alpha_short"],
        "alpha_long": best["alpha_long"],
        "N_grid": N_grid.tolist(),
        "alpha_grid": alpha_grid.tolist(),
    }

    oof_metrics = {"oof_f1": best["f1"], "oof_roc_auc": best["roc_auc"], "oof_accuracy": best["accuracy"]}

    log_experiment(
        exp_name=exp_name,
        model="conditional_blend(TFIDF,DistilBERT)",
        seed=seed,
        n_splits=n_splits,
        params=params,
        threshold=threshold,
        oof_metrics=oof_metrics,
        test_metrics=test_metrics,
        notes="Conditional blending by text length (word count), params selected on OOF F1",
    )


if __name__ == "__main__":
    main()